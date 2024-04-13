/*
use anyhow::{anyhow, bail, Result};
use audio::BufMut;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::Resampler;
use rubato::{SincInterpolationParameters, SincInterpolationType, SincFixedIn, WindowFunction};
use std::fs;
use std::io;
use std::path::Path;

const CHUNK_SIZE: usize = 1024;
pub fn run_example() -> Result<()> {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("failed to build default device"))?;

    let config = device.default_output_config()?;

    let _dag_nasty_circles = "E:/Mp3s/20 Years of Dischord/20 Years of Dischord Disk 1/120-20_years_of_dishord-20_dag_nasty-circles.mp3";
    let _waiting_room = "E:/Mp3s/Fugazi/Fugazi - 1988 - 13 Songs/01.waiting_room.mp3";
    let _eggy_toast = "E:/New Bands/Eggy Toast - Lose your head.mp3.mp3";
    let _hotsnakes_having_another = "E:/Mp3s/Hot Snakes/hot_snakes-jericho_sirens-320k/sp1224-07_having_another.mp3";
    let _wmut = "E:/Mp3s/We Meet Under Tables/Demos/We Meet Under Tables - Greater Minds.mp3";

    let path = Path::new(_dag_nasty_circles); 

    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(path, &device, &config),
        cpal::SampleFormat::I16 => run::<i16>(path, &device, &config),
        cpal::SampleFormat::U16 => run::<u16>(path, &device, &config),
        _ => panic!("unsupported sample_format")
    }
}

fn run<T>(path: &Path, device: &cpal::Device, config: &cpal::SupportedStreamConfig) -> Result<()>
where
    T: 'static + Send + cpal::Sample + cpal::SizedSample + audio::Sample + audio::Translate<f32>,
    f32: audio::Translate<i16>,
{
    let source = io::BufReader::new(fs::File::open(path)?);
    let decoder = minimp3::Decoder::new(source);

    let config = cpal::StreamConfig {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let pcm = audio::buf::Interleaved::new();
    let output = audio::buf::Interleaved::with_topology(config.channels as usize, 1024);
    let resample = audio::buf::Sequential::with_topology(config.channels as usize, CHUNK_SIZE);

    let mut writer = Writer {
        frames: 0,
        seconds: 0.0,
        decoder,
        pcm: audio::io::Read::new(pcm),
        resampler: None,
        output: audio::io::ReadWrite::empty(output),
        resample: audio::io::ReadWrite::empty(resample),
        device_sample_rate: config.sample_rate.0,
        device_channels: config.channels as usize,
        last_frame: None,
    };

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            if let Err(e) = writer.write_to(data) {
                println!("failed to write data: {}", e);
            }
        },
        move |err| {
            println!("audio output error: {}", err);
        },
        None
    )?;

    stream.play()?;

    let mut line = String::new();

    println!("press [enter] to shut down");
    std::io::stdin().read_line(&mut line)?;
    Ok(())
}

struct Writer<R>
where
    R: io::Read,
{
    // Frame counter.
    frames: usize,
    // Last second counter.
    seconds: f32,
    // The open mp3 decoder.
    decoder: minimp3::Decoder<R>,
    // Buffer used for mp3 decoding.
    pcm: audio::io::Read<audio::buf::Interleaved<i16>>,
    // The last mp3 frame decoded.
    last_frame: Option<minimp3::Frame>,
    // Sampler that is used in case the sample rate of a decoded frame needs to
    // be resampled.
    resampler: Option<rubato::SincFixedIn<f32>>,
    // Output buffer to flush to device buffer.
    output: audio::io::ReadWrite<audio::buf::Interleaved<f32>>,
    // Resample buffer.
    resample: audio::io::ReadWrite<audio::buf::Sequential<f32>>,
    // Sample rate expected to be written to the device.
    device_sample_rate: u32,
    // Number of channels in the device.
    device_channels: usize,
}

impl<R> Writer<R>
where
    R: io::Read,
{
    // The decoder loop.
    fn write_to<T>(&mut self, out: &mut [T]) -> anyhow::Result<()>
    where
        T: 'static + Send + audio::Sample + audio::Translate<f32> + cpal::SizedSample,
    {
        use audio::{io, wrap};
        use audio::{Buf, ExactSizeBuf, ReadBuf, WriteBuf};

        let mut data = io::Write::new(wrap::interleaved(out, self.device_channels));

        // Run the loop while there is buffer to fill.
        while data.has_remaining_mut() {
            // If there is output available, translate it to the data buffer
            // used by cpal.
            //
            // cpal's data buffer expects the output to be interleaved.
            if self.output.has_remaining() {
                tracing::trace! {
                    remaining = self.output.remaining(),
                    remaining_mut = data.remaining_mut(),
                    "translate remaining",
                };
                io::translate_remaining(&mut self.output, &mut data);
                continue;
            }

            tracing::trace!(remaining_mut = data.remaining_mut(), "writing data");

            // If we have collected exactly one CHUNK_SIZE of resample buffer,
            // process it through the resampler and translate its result to the
            // output buffer.
            if let Some(frame) = &self.last_frame {
                if self.resample.remaining() == CHUNK_SIZE {
                    let device_sample_rate = self.device_sample_rate;

                    let resampler = self.resampler.get_or_insert_with(|| {
                        let params = SincInterpolationParameters {
                            sinc_len: 256,
                            f_cutoff: 0.95,
                            interpolation: SincInterpolationType::Linear,
                            oversampling_factor: 256,
                            window: WindowFunction::BlackmanHarris2,
                        };

                        let f_ratio = device_sample_rate as f64 / frame.sample_rate as f64;
                        SincFixedIn::<f32>::new(
                            f_ratio,
                            2.,
                            params,
                            CHUNK_SIZE,
                            frame.channels as usize,
                        ).expect("failed to create resampler")
                    });

                    tracing::trace!("processing resample chunk");

                    // let wi = self.resample.into_inner().into_vec();
                    // let mut wo = self.output.into_inner().into_vec();

                    // resampler.process_into_buffer(
                    //     &wi,
                    //     &mut wo,
                    //     None,// &bittle::all(),
                    // )?;


                    resampler.process_into_buffer(
                        self.resample.as_ref(),
                        self.output.as_mut(),
                        None,// &bittle::all(),
                    )?;

                    // resampler.process(
                    //     self.resample.as_ref(),
                    //     // self.output.as_mut(),
                    //     &bittle::all(),
                    // )?;

                    self.resample.clear();
                    let frames = self.output.as_ref().frames();

                    self.output.set_read(0);
                    self.output.set_written(frames);
                    continue;
                }
            }

            // If we have information on a decoded frame, translate it into the
            // resample buffer until its filled up to its frames cap.
            if self.pcm.has_remaining() {
                tracing::trace! {
                    remaining_mut = self.pcm.remaining(),
                    remaining = self.resample.remaining_mut(),
                    "translate remaining from pcm to resample buffer",
                };
                io::translate_remaining(&mut self.pcm, &mut self.resample);
                continue;
            }

            tracing::trace! {
                frames = self.pcm.as_ref().frames(),
                "decode frame",
            };

            let frame = self.decoder.next_frame()?;
            self.pcm.set_read(0);
            self.resample.as_mut().resize_channels(frame.channels);

            // If the sample rate of the decoded frames matches the expected
            // output exactly, copy it directly to the output frame without
            // resampling.
            if frame.sample_rate as u32 == self.device_sample_rate {
                tracing::trace! {
                    remaining_mut = self.pcm.remaining(),
                    remaining = self.resample.remaining_mut(),
                    "translate remaining from pcm directly to output",
                };
                io::translate_remaining(&mut self.pcm, &mut self.output);
                continue;
            }

            self.last_frame = Some(frame);
        }

        // If the last decoded frame contains fewer channels than we have data
        // channels, copy channel 0 to the first two presumed stereo channels.
        if let Some(frame) = &self.last_frame {
            if frame.channels == 0 {
                bail!("tried to play stream with zero channels")
            }

            for to in frame.channels..usize::min(data.channels(), 2) {
                data.copy_channel(0, to);
            }
        }

        self.frames = self.frames.saturating_add(data.as_ref().frames());

        let seconds = self.frames as f32 / self.device_sample_rate as f32;
        let s = seconds.floor();

        if s > self.seconds {
            use std::io::Write as _;
            let mut o = std::io::stdout();
            write!(
                o,
                "\rTime: {:02}:{:02}",
                (s / 60.0) as u32,
                (s % 60.0) as u32
            )?;
            o.flush()?;
            self.seconds = seconds;
        }

        Ok(())
    }
}
*/
