#![allow(dead_code)]
use cpal::{Sample, SampleFormat};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::io::Seek;
use std::sync::mpsc::{Sender};
use std::vec;

use minimp3::{Decoder, Frame};

use std::fs::File;
use std::io::{Read};
// use rodio::Decoder as RodioDecoder;
// use rodio::source::Source;

use rubato::{Resampler, SincFixedIn, InterpolationType, InterpolationParameters, WindowFunction};

fn main() {
    //config_stuff();
    play_some_audio();

    println!("Done");
}

fn config_stuff() {
    let host = cpal::default_host();
    let input_devices = host.input_devices();
    let output_devices = host.output_devices();

    println!("Input Devices\n===============");
    for input_device in input_devices {
        for device in input_device {
            println!("name: {}", device.name().expect("coulnd't get name of input device"));
        }
    }

    println!("\nOutput Devices\n===============");
    for output_device in output_devices {
        for device in output_device {
            println!("name: {}", device.name().expect("coulnd't get name of output device"));

            for cfgs in device.supported_input_configs() {
                for cfg in cfgs {
                    println!(
                        "\t input_configs => channels: {}, buffer size: {:?}, sample_format: {:?}, min_sample_rate: {:?}, max_sample_rate: {:?}",
                        &cfg.channels(),
                        &cfg.buffer_size(),
                        &cfg.sample_format(),
                        &cfg.min_sample_rate(),
                        &cfg.max_sample_rate(),
                    );
                }
            }

            for cfgs in device.supported_output_configs() {
                for cfg in cfgs {
                    println!(
                        "\t output_configs => channels: {}, buffer size: {:?}, sample_format: {:?}, min_sample_rate: {:?}, max_sample_rate: {:?}",
                        &cfg.channels(),
                        &cfg.buffer_size(),
                        &cfg.sample_format(),
                        &cfg.min_sample_rate(),
                        &cfg.max_sample_rate(),
                    );
                }
            }
        }
    }
}

fn play_some_audio() {
    // let (sender, receiver) = channel();

    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();

    let mut supported_configs_range = device.supported_output_configs().expect("Error querying output config");
    let supported_config = supported_configs_range.next().expect("no output support config!?")
        .with_max_sample_rate();
    // .with_sample_rate(cpal::SampleRate(44100));

    // println!("min sample rate {:?}", supported_config.min_sample_rate());
    // println!("max sample rate {:?}", supported_config.max_sample_rate());
    
    let output_err_fn = |err| eprintln!("an error occurred on the output audio stream {}", err);
    
    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();

    // Produce sinusoidal wave
    let sample_rate = config.sample_rate.0 as f32;

    println!("config sample rate: {}", sample_rate);

    // let channels = config.channels as usize;
    // let mut sample_clock = 0f32;
    // let mut next_value = move || {
    //     sample_clock = (sample_clock + 1.0) % sample_rate;
    //     (sample_clock * 440.0 * 2.0 * std::f32::consts::PI / sample_rate).sin()
    // };

    // play mp3
    let _dag_nasty_circles = "E:/Mp3s/20 Years of Dischord/20 Years of Dischord Disk 1/120-20_years_of_dishord-20_dag_nasty-circles.mp3";
    let _waiting_room = "E:/Mp3s/Fugazi/Fugazi - 1988 - 13 Songs/01.waiting_room.mp3";
    let _eggy_toast = "E:/New Bands/Eggy Toast - Lose your head.mp3.mp3";
    let _hotsnakes_having_another = "E:/Mp3s/Hot Snakes/hot_snakes-jericho_sirens-320k/sp1224-07_having_another.mp3";
    let _wmut = "E:/Mp3s/We Meet Under Tables/Demos/We Meet Under Tables - Greater Minds.mp3";

    // let mut decoder = Decoder::new(File::open(&dag_nasty_circles).unwrap());
    let buf = std::io::BufReader::new(File::open(&_hotsnakes_having_another).unwrap());
    let  mp3_decoder = Mp3Decoder::new(buf).expect("failed to create mp3 decoder");

    let mut all_mp3_samples_f64 = mp3_decoder.map(|s| s as f64).collect::<Vec<f64>>();    
    let mut left: Vec<f64> = vec![];
    let mut right: Vec<f64> = vec![];

    for (idx, sample) in all_mp3_samples_f64.iter().enumerate() {
        if idx % 2 == 0 {
            left.push(*sample);
        } else {
            right.push(*sample);
        }
    }

    assert!(left.len() == right.len());

    println!("About to setup resample parameters");
    
    let params = InterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    
    let mut resampler = SincFixedIn::<f64>::new(
        48000 as f64 / 44100 as f64,
        2.0,
        params,
        left.len(),
        2
    ).unwrap();
    
    println!("About to resample");
    
    let channels = vec![left, right];
    let resampled_audio = resampler.process(&channels, None).expect("Failed to resample");


    println!("Just finished resampling");

    let zip_audio = resampled_audio[0]
        .iter()
        .zip(&resampled_audio[1])
        .collect::<Vec<(&f64, &f64)>>();

    let mut vec_channels = vec![];
    for z in zip_audio {
        vec_channels.push(vec![*z.0, *z.1]);
    }

    let mut flat_channels = vec_channels
        .iter()
        .flatten()
        .map(|s| *s as i16)
        .collect::<Vec<i16>>()
        .into_iter();


    let mut next_sample = move || {  
        match flat_channels.next() {
            Some(sample) => sample,
            None => 0i16,
        }
    };


    // let mut next_sample = move || {  
    //     match mp3_decoder.next() {
    //         Some(sample) => sample,
    //         None => 0i16,
    //     }
    // };
    
    let output_stream = match sample_format {
        SampleFormat::F32 => device.build_output_stream(
            &config, 
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| write_sample(data, &mut next_sample),
            output_err_fn),
        SampleFormat::I16 => device.build_output_stream(
            &config, 
            move |data: &mut [i16], _: &cpal::OutputCallbackInfo| write_sample(data, &mut next_sample),
            output_err_fn),
        SampleFormat::U16 => device.build_output_stream(
            &config, 
            move |data: &mut [u16], _: &cpal::OutputCallbackInfo| write_sample(data, &mut next_sample),
        output_err_fn),
        // _ => panic!("sample format unknown")
    }.unwrap();

    output_stream.play().unwrap();

    // let tx = sender.clone();

    // let input_err_fn = |err| eprintln!("an error occurred on the input audio stream {}", err);
    
    // let input_stream = match sample_format {
    //     SampleFormat::F32 => device.build_input_stream(
    //         &config, 
    //         move |data: &[f32], _: &cpal::InputCallbackInfo| process_incoming_data(data, channels, &tx),
    //         input_err_fn),
    //     SampleFormat::I16 => device.build_input_stream(
    //         &config, 
    //         move |data: &[i16], _: &cpal::InputCallbackInfo| process_incoming_data(data, channels, &tx),
    //         input_err_fn),
    //     SampleFormat::U16 => device.build_input_stream(
    //         &config, 
    //         move |data: &[u16], _: &cpal::InputCallbackInfo| process_incoming_data(data, channels, &tx),
    //         input_err_fn),
    // }.unwrap();

    
    // output_stream.play().unwrap();
    // input_stream.play().unwrap();

    // for i in 0..350_000 {
    //     match receiver.try_recv() {
    //         Ok(data) => println!("From the incoming audio: {} <<<>>> {}", i, data),
    //         Err(err) => (),//println!("ERR RECEIVING: {}", err),
    //     }
    // }

    println!("Sleeping");
    std::thread::sleep(std::time::Duration::from_secs(120));
    println!("Done!");
}

fn write_mp3_frame<T: Sample>(data: &mut [T], next_frame: &mut dyn FnMut() -> Frame) 
    where T: cpal::Sample 
{
    let mp3_frame = next_frame();
    let mp3_data = mp3_frame.data;
    let _mp3_channels = mp3_frame.channels;
    // let sample_rate = mp3_frame.sample_rate;

    let mut samples = vec![];
    // for incoming_frame in mp3_data.chunks(mp3_channels) {
    //     for item in incoming_frame {
    //         let value: T = cpal::Sample::from::<i16>(item);
    //         samples.push(value);
    //     }
    // }

    for incoming_frame in mp3_data {
        let value: T = cpal::Sample::from::<i16>(&incoming_frame);
        samples.push(value);
    }

    for (source, destination) in samples.iter().zip(data.iter_mut()) {
        *destination = *source;
    }
}

fn write_sample<T: cpal::Sample>(data: &mut [T], next_sample: &mut dyn FnMut() -> i16)
{
    for frame in data.chunks_mut(1) {
        let value: T = cpal::Sample::from::<i16>(&next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }        
}

// fn write_sample<T: cpal::Sample>(data: &mut [T], next_sample: &mut dyn FnMut() -> f32)
// {
//     for frame in data.chunks_mut(1) {
//         let value: T = cpal::Sample::from::<f32>(&next_sample());
//         for sample in frame.iter_mut() {
//             *sample = value;
//         }
//     }        
// }

fn process_incoming_data<T: cpal::Sample>(data: &[T], _channels: usize, tx: &Sender<f32>) 
{
    for sample in data.iter() {
        tx.send(sample.to_f32()).expect("failed sending the incoming stream");
    }
}

pub struct Mp3Decoder<R>
where
    R: Read + Seek
{
    decoder: Decoder<R>,
    current_frame: Frame,
    current_frame_offset: usize,
}


impl<R> Mp3Decoder<R>
where
    R: Read + Seek
{
    pub fn new(data: R) -> Result<Self, ()> {
        let mut decoder = Decoder::new(data);
        let current_frame = decoder.next_frame().map_err(|_| ())?;

        Ok(
            Self {
                decoder,
                current_frame,
                current_frame_offset: 0,
            }
        )
    }
}

impl<R> Iterator for Mp3Decoder<R>
where
    R: Read + Seek
{
    type Item = i16;

    #[inline]
    fn next(&mut self) -> Option<i16> {
        if self.current_frame_offset == self.current_frame.data.len() {
            match self.decoder.next_frame() {
                Ok(frame) => self.current_frame = frame,
                _ => return None,
            }
            self.current_frame_offset = 0;
        }

        let v = self.current_frame.data[self.current_frame_offset];
        self.current_frame_offset += 1;

        Some(v)
    }
}
