extern crate sdl2;
use anyhow::{anyhow, bail, Result};
use realfft::RealFftPlanner;
use sdl2::audio::{AudioCallback, AudioSpecDesired};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use std::f32::consts::PI;
use std::mem::MaybeUninit;
use std::sync::Arc;

use ringbuf::{Consumer, HeapRb, Producer, SharedRb};

const WAVETABLE_SAMPLE_COUNT: usize = 512;
const MIN_FREQ: f32 = 20.0;
const WAVETABLE_SIZE: usize = 4096;

pub struct Wavetable {
    waveform: Waveform,
    wavetable: Vec<WavetableEntry>,
}

impl Wavetable {
    pub fn new(waveform: Waveform, sample_rate: f32) -> Self {
        let nyquist = sample_rate / 2.0;
        let mut freq = MIN_FREQ;
        let table_entry_len = (nyquist / MIN_FREQ).log2() as usize + 1;
        let mut wavetable: Vec<WavetableEntry> = Vec::with_capacity(table_entry_len);

        for n in 0..table_entry_len {
            wavetable.push(WavetableEntry::new(sample_rate, freq));
            freq *= 2.0;
        }

        Self {
            waveform,
            wavetable,
        }
    }

    /* TODO - can I return a reference to the actual wavtable instead of index? */
    pub fn tbl_index(&self, freq: f32) -> usize {
        for (i, wt) in self.wavetable.iter().enumerate() {
            /* Linear search for wavetable entry. Improve with a Binary Search */
            if freq < wt.max_freq {
                return i;
            }
        }

        // use last table if freq is too high
        self.wavetable.len() - 1
    }
}

pub struct WavetableEntry {
    pub max_freq: f32,
    pub buffer: Vec<f32>,
}

impl WavetableEntry {
    pub fn new(sample_rate: f32, freq: f32) -> Self {
        Self {
            max_freq: freq,
            buffer: WavetableEntry::generate_buffer(sample_rate, freq),
        }
    }

    pub fn wavetable_eval(&self, phs: f32) -> f32 {
        if phs < 0.0 || phs >= 1.0 {
            return 0.0;
        }

        let table_value = phs * WAVETABLE_SIZE as f32;
        let i = table_value.floor() as usize;
        let fract = table_value.fract() as f32;
        let x0 = self.buffer[i];
        let x1 = self.buffer[(i + 1) % WAVETABLE_SIZE];
        (1.0 - fract) * x0 + fract * x1 /* weighted LERP between the two samples */
    }

    // For Sawtooth Fourier Coefficient
    fn coeff(k: f32) -> f32 {
        if k % 2.0 == 0.0 {
            -1.0 / k
        } else {
            1.0 / k
        }
    }

    pub fn generate_buffer(sample_rate: f32, freq: f32) -> Vec<f32> {
        let nyquist = sample_rate / 2.0;
        let step = 2.0 * PI / WAVETABLE_SIZE as f32;
        let mut max = 0.0;
        let mut buffer: Vec<f32> = Vec::with_capacity(WAVETABLE_SIZE);

        /* TODO - use vec![] or another rusty way to do this */
        for _ in 0..WAVETABLE_SIZE {
            buffer.push(0.0);
        }

        for i in 0..WAVETABLE_SIZE {
            let phi = i as f32 * step;

            /* Add up harmonics */
            buffer[i] = 0.0;
            for k in 1..(nyquist as usize) {
                buffer[i] += WavetableEntry::coeff(k as f32) * (k as f32 * phi).sin();
                if k as f32 * freq >= nyquist {
                    break;
                }
            }

            /* Update max amplitude for normalization */
            if buffer[i].abs() > max {
                max = buffer[i].abs();
            }
        }

        /* Normalize amplitude -1..1 */
        for i in 0..WAVETABLE_SIZE {
            if max > 0.0 {
                buffer[i] /= max;
            }
        }

        buffer
    }
}

#[derive(Copy, Clone)]
enum Waveform {
    Sine,
    Square,
    Triangle,
    Sawtooth,
    SineWavetable,
    SawtoothWavetable,
}

struct AudioContext {
    volume: f32,
    freq: f32,
    is_playing: bool,
    waveform: Waveform,
}

struct Oscillator {
    ctx: Consumer<AudioContext, Arc<SharedRb<AudioContext, Vec<MaybeUninit<AudioContext>>>>>,
    audio_out: Producer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>,
    sample_rate: f32,
    i: f32,
    phasor: f32,
    sin_wavetable: Vec<f32>,
    wavetable: Wavetable,
}

impl Oscillator {
    pub fn wavetable_eval(&self, phs: f32) -> f32 {
        if phs < 0.0 || phs >= 1.0 {
            return 0.0;
        }

        let table_value = phs * WAVETABLE_SAMPLE_COUNT as f32;
        let i = table_value.floor() as usize;
        let fract = table_value.fract() as f32;
        let x0 = self.sin_wavetable[i];
        let x1 = self.sin_wavetable[(i + 1) % WAVETABLE_SAMPLE_COUNT];
        (1.0 - fract) * x0 + fract * x1 /* weighted LERP between the two samples */
    }
}

impl AudioCallback for Oscillator {
    type Channel = f32;

    // Don't allocate here
    fn callback(&mut self, out: &mut [f32]) {
        if let Some(ctx) = self.ctx.pop() {
            for x in out.iter_mut() {
                /*
                // This formula uses individual sample, but a better one is to use phasor.
                *x = (2.0 * PI * ctx.freq * (self.i / self.sample_rate)).sin() * ctx.volume;
                self.i += 1.0;
                */

                self.phasor += ctx.freq / self.sample_rate;

                // Clamp the phasor [0, 1);
                while self.phasor >= 1.0 {
                    self.phasor -= 1.0;
                }
                while self.phasor < 0.0 {
                    self.phasor += 1.0;
                }

                *x = match ctx.waveform {
                    Waveform::Sine => (2.0 * PI * self.phasor).sin() * ctx.volume,
                    Waveform::Square => {
                        if self.phasor < 0.5 {
                            -1.0 * ctx.volume
                        } else {
                            1.0 * ctx.volume
                        }
                    }
                    Waveform::Triangle => {
                        if self.phasor < 0.5 {
                            (4.0 * self.phasor - 1.0) * ctx.volume
                        } else {
                            ((4.0 * (1.0 - self.phasor)) - 1.0) * ctx.volume
                        }
                    }
                    Waveform::Sawtooth => (2.0 * self.phasor - 1.0) * ctx.volume,
                    Waveform::SineWavetable => self.wavetable_eval(self.phasor) * ctx.volume,
                    Waveform::SawtoothWavetable => {
                        let freq = ctx.freq;
                        let idx = self.wavetable.tbl_index(freq);
                        let wt_entry = &self.wavetable.wavetable[idx];
                        let buffer = &wt_entry.buffer;
                        wt_entry.wavetable_eval(self.phasor) * ctx.volume
                    }
                };
            }
        } else {
            for x in out.iter_mut() {
                *x = 0.0;
            }
        }

        let _ = self.audio_out.push_slice(out);
    }
}

const SCREEN_WIDTH: i32 = 2048;
const SCREEN_HEIGHT: i32 = 512;
const RING_BUF_SIZE: usize = 2048;

pub fn run() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let audio_subsystem = sdl_context.audio()?;
    let sample_rate = 44_100.0;

    let mut real_planner = RealFftPlanner::<f32>::new();

    let desired_spec = AudioSpecDesired {
        freq: Some(sample_rate as i32),
        channels: Some(1),
        samples: None,
    };

    let ring_buf = HeapRb::<AudioContext>::new(2);
    let (mut prod, cons) = ring_buf.split();

    let audio_out_buf = HeapRb::<f32>::new(RING_BUF_SIZE);
    let (prod_out, mut cons_out) = audio_out_buf.split();

    let device = audio_subsystem.open_playback(None, &desired_spec, |spec| {
        println!("{:?}", spec);

        let phase_angle_step = 1.0f32 / WAVETABLE_SAMPLE_COUNT as f32;
        let sin_wavetable = (0..WAVETABLE_SAMPLE_COUNT)
            .map(|i| (2.0 * PI * i as f32 * phase_angle_step).sin())
            .collect::<Vec<f32>>();
        let saw_wavetable = Wavetable::new(Waveform::Sawtooth, sample_rate);

        Oscillator {
            ctx: cons,
            audio_out: prod_out,
            i: 0.0,
            sample_rate,
            phasor: 0.0,
            sin_wavetable,
            wavetable: saw_wavetable,
        }
    })?;

    device.resume();

    let window = video_subsystem
        .window("synth_demo", SCREEN_WIDTH as u32, SCREEN_HEIGHT as u32)
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window
        .into_canvas()
        .software()
        .build()
        .map_err(|e| e.to_string())?;

    let mut prev_buttons = std::collections::HashSet::new();
    let mut color = Color::RGB(0, 0, 0);

    canvas.set_draw_color(color);
    canvas.clear();
    canvas.present();

    let mut event_pump = sdl_context.event_pump()?;
    let mut waveform = Waveform::Sine;
    let mut scope = (0..2048).map(|_| 0.0).collect::<Vec<f32>>();

    'running: loop {
        canvas.set_draw_color(color);
        canvas.clear();

        for event in event_pump.poll_iter() {
            match event {
                Event::KeyDown {
                    keycode: Some(Keycode::A),
                    ..
                } => waveform = Waveform::Sine,
                Event::KeyDown {
                    keycode: Some(Keycode::S),
                    ..
                } => waveform = Waveform::Square,
                Event::KeyDown {
                    keycode: Some(Keycode::D),
                    ..
                } => waveform = Waveform::Triangle,
                Event::KeyDown {
                    keycode: Some(Keycode::F),
                    ..
                } => waveform = Waveform::Sawtooth,
                Event::KeyDown {
                    keycode: Some(Keycode::G),
                    ..
                } => waveform = Waveform::SineWavetable,
                Event::KeyDown {
                    keycode: Some(Keycode::H),
                    ..
                } => waveform = Waveform::SawtoothWavetable,
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        let mouse_state = event_pump.mouse_state();
        let buttons: std::collections::HashSet<_> = mouse_state.pressed_mouse_buttons().collect();

        if buttons.contains(&MouseButton::Left) && prev_buttons.contains(&MouseButton::Left) {
            let volume = mouse_state.y() as f32 / SCREEN_WIDTH as f32;
            let freq = ((mouse_state.x() as f32 / SCREEN_HEIGHT as f32) * 1000.) + 20.;
            let new_ctx = AudioContext {
                freq,
                volume,
                is_playing: true,
                waveform,
            };
            _ = prod.push(new_ctx);

            //let color_inc = mouse_state.x() as u8;
            //color = Color::RGB(color.r + color_inc, color.g + color_inc, color.b + color_inc);
        } else {
            let new_ctx = AudioContext {
                freq: 440.0,
                volume: 0.0,
                is_playing: false,
                waveform: Waveform::Sine,
            };
            _ = prod.push(new_ctx);
        }

        let _out_len = cons_out.pop_slice(&mut scope[..]);
        let fft_len = RING_BUF_SIZE;
        let r2c = real_planner.plan_fft_forward(fft_len);
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut scope[..], &mut spectrum)
            .expect("failed to process FFT");

        canvas.set_draw_color(Color::RGB(100, 255, 100));

        for (x, f) in spectrum.iter().enumerate() {
            _ = canvas.draw_point(Point::new(
                x as i32,
                (((SCREEN_HEIGHT / 3) as f32 * 2.0) - f.re.abs()) as i32,
            ));
        }

        for (x, sample) in scope.iter().enumerate() {
            _ = canvas.draw_point(Point::new(
                x as i32,
                ((SCREEN_HEIGHT / 3) as f32 - (sample * 100.0)) as i32,
            ));
        }
        canvas.present();
        prev_buttons = buttons;
    }

    Ok(())
}
