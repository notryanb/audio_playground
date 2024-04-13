extern crate sdl2;
use anyhow::{anyhow, bail, Result};
use sdl2::audio::{AudioCallback, AudioSpecDesired};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::Color;
use std::mem::MaybeUninit;
use std::sync::Arc;

use ringbuf::{Consumer, HeapRb, SharedRb};

struct AudioContext {
    volume: f32,
    freq: f32,
    is_playing: bool,
}

struct Oscillator {
    ctx: Consumer<AudioContext, Arc<SharedRb<AudioContext, Vec<MaybeUninit<AudioContext>>>>>,
    sample_rate: f32,
    i: f32,
}

impl AudioCallback for Oscillator {
    type Channel = f32;

    // Don't allocate here
    fn callback(&mut self, out: &mut [f32]) {
        if let Some(ctx) = self.ctx.pop() {
            for x in out.iter_mut() {
                *x = (2.0 * std::f32::consts::PI * ctx.freq * (self.i / self.sample_rate)).sin()
                    * ctx.volume;
                self.i += 1.0;
            }
        } else {
            for x in out.iter_mut() {
                *x = 0.0;
            }
        }
    }
}

const SCREEN_WIDTH: i32 = 512;
const SCREEN_HEIGHT: i32 = 512;

pub fn run() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let audio_subsystem = sdl_context.audio()?;

    let desired_spec = AudioSpecDesired {
        freq: Some(44_100),
        channels: Some(1),
        samples: None,
    };

    let ring_buf = HeapRb::<AudioContext>::new(2);
    let (mut prod, cons) = ring_buf.split();

    let device = audio_subsystem.open_playback(None, &desired_spec, |spec| {
        println!("{:?}", spec);

        Oscillator {
            ctx: cons,
            i: 0.0,
            sample_rate: 44_100.0,
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

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
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
            let freq = ((mouse_state.x() as f32 / SCREEN_HEIGHT as f32) * 4950.) + 50.;
            let new_ctx = AudioContext {
                freq,
                volume,
                is_playing: true,
            };
            _ = prod.push(new_ctx);

            //let color_inc = mouse_state.x() as u8;
            //color = Color::RGB(color.r + color_inc, color.g + color_inc, color.b + color_inc);
        } else {
            let new_ctx = AudioContext {
                freq: 440.0,
                volume: 0.0,
                is_playing: false,
            };
            _ = prod.push(new_ctx);
        }

        canvas.set_draw_color(color);
        canvas.clear();
        canvas.present();
        prev_buttons = buttons;
    }

    Ok(())
}