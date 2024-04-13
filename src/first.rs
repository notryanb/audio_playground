// #![allow(dead_code)]
// use cpal::{FromSample, Sample, SampleFormat, SizedSample};
// use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
// use std::io::{Seek,Read,Write};
// use std::sync::mpsc::{Sender};
// use std::vec;

// use minimp3::{Decoder, Frame};

// use std::fs::File;
// // use rodio::Decoder as RodioDecoder;
// // use rodio::source::Source;

// use rubato::{
//     Resampler, SincFixedIn,
//     SincInterpolationType, SincInterpolationParameters, 
//     WindowFunction
// };


// fn queue() {
//     let _dag_nasty_circles = "E:/Mp3s/20 Years of Dischord/20 Years of Dischord Disk 1/120-20_years_of_dishord-20_dag_nasty-circles.mp3";
//     let _waiting_room = "E:/Mp3s/Fugazi/Fugazi - 1988 - 13 Songs/01.waiting_room.mp3";
//     let _eggy_toast = "E:/New Bands/Eggy Toast - Lose your head.mp3.mp3";
//     let _hotsnakes_having_another = "E:/Mp3s/Hot Snakes/hot_snakes-jericho_sirens-320k/sp1224-07_having_another.mp3";
//     let _wmut = "E:/Mp3s/We Meet Under Tables/Demos/We Meet Under Tables - Greater Minds.mp3";
// }

// // struct AudioQueue {
// //     tracks: Vec<
// // }

// fn config_stuff() {
//     let host = cpal::default_host();
//     let input_devices = host.input_devices();
//     let output_devices = host.output_devices();

//     println!("Input Devices\n===============");
//     for input_device in input_devices {
//         for device in input_device {
//             println!("name: {}", device.name().expect("coulnd't get name of input device"));
//         }
//     }

//     println!("\nOutput Devices\n===============");
//     for output_device in output_devices {
//         for device in output_device {
//             println!("name: {}", device.name().expect("coulnd't get name of output device"));

//             for cfgs in device.supported_input_configs() {
//                 for cfg in cfgs {
//                     println!(
//                         "\t input_configs => channels: {}, buffer size: {:?}, sample_format: {:?}, min_sample_rate: {:?}, max_sample_rate: {:?}",
//                         &cfg.channels(),
//                         &cfg.buffer_size(),
//                         &cfg.sample_format(),
//                         &cfg.min_sample_rate(),
//                         &cfg.max_sample_rate(),
//                     );
//                 }
//             }

//             for cfgs in device.supported_output_configs() {
//                 for cfg in cfgs {
//                     println!(
//                         "\t output_configs => channels: {}, buffer size: {:?}, sample_format: {:?}, min_sample_rate: {:?}, max_sample_rate: {:?}",
//                         &cfg.channels(),
//                         &cfg.buffer_size(),
//                         &cfg.sample_format(),
//                         &cfg.min_sample_rate(),
//                         &cfg.max_sample_rate(),
//                     );
//                 }
//             }
//         }
//     }
// }

// fn read_frames<R: Read + Seek>(input_buffer: &mut R, nbr: usize, channels: usize) -> Vec<Vec<f32>> {
//     let mut buffer = vec![0u8; 4];
//     let mut wfs = Vec::with_capacity(channels);
//     for _chan in 0..channels {
//        wfs.push(Vec::with_capacity(nbr)); 
//     }
//     let mut value: f32;
//     for _frame in 0..nbr {
//         for wf in wfs.iter_mut().take(channels) {
//             input_buffer.read(&mut buffer).unwrap();
//             value = f32::from_le_bytes(buffer.as_slice().try_into().unwrap()) as f32;
//             wf.push(value);
//         }
//     }

//     wfs
// }

// fn write_frames<W: Write + Seek>(waves: Vec<Vec<f32>>, output_buffer: &mut W, channels: usize) {
//     let nbr = waves[0].len();
//     for frame in 0..nbr {
//         for chan in 0..channels {
//             let value = waves[chan][frame];
//             let bytes = value.to_le_bytes();
//             output_buffer.write(&bytes).unwrap();
//         }
//     }
// }

// fn play_some_audio_sync() {
//     let host = cpal::default_host();
//     let device = host.default_output_device().unwrap();

//     println!("Device: {}", device.name().expect("expected a device name..."));

//     let mut supported_configs_range = device.supported_output_configs().expect("Error querying output config");
//     let supported_config = supported_configs_range
//         .next()
//         .expect("no output support config!?")
//         .with_max_sample_rate();

//     let sample_format = supported_config.sample_format();
//     let config: cpal::StreamConfig = supported_config.into();

//     match sample_format {
//         cpal::SampleFormat::I8 => run::<i8>(&device, &config.into()),
//         cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()),
//         // cpal::SampleFormat::I24 => run::<I24>(&device, &config.into()),
//         cpal::SampleFormat::I32 => run::<i32>(&device, &config.into()),
//         // cpal::SampleFormat::I48 => run::<I48>(&device, &config.into()),
//         cpal::SampleFormat::I64 => run::<i64>(&device, &config.into()),
//         cpal::SampleFormat::U8 => run::<u8>(&device, &config.into()),
//         cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()),
//         // cpal::SampleFormat::U24 => run::<U24>(&device, &config.into()),
//         cpal::SampleFormat::U32 => run::<u32>(&device, &config.into()),
//         // cpal::SampleFormat::U48 => run::<U48>(&device, &config.into()),
//         cpal::SampleFormat::U64 => run::<u64>(&device, &config.into()),
//         cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()),
//         cpal::SampleFormat::F64 => run::<f64>(&device, &config.into()),
//         sample_format => panic!("Unsupported sample format '{sample_format}'"),
//     };

//     // println!("Sleeping");
//     // std::thread::sleep(std::time::Duration::from_secs(120));
//     // println!("Done!");
// }

// pub fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) 
//     where T: SizedSample + FromSample<i16>
// {
//     let device_sample_rate = config.sample_rate.0;

//     println!("config sample rate: {}", device_sample_rate);

//     let err_fn = |err| eprintln!("an error occurred on the output audio stream {}", err);

//     // play mp3
//     let _dag_nasty_circles = "E:/Mp3s/20 Years of Dischord/20 Years of Dischord Disk 1/120-20_years_of_dishord-20_dag_nasty-circles.mp3";
//     let _waiting_room = "E:/Mp3s/Fugazi/Fugazi - 1988 - 13 Songs/01.waiting_room.mp3";
//     let _eggy_toast = "E:/New Bands/Eggy Toast - Lose your head.mp3.mp3";
//     let _hotsnakes_having_another = "E:/Mp3s/Hot Snakes/hot_snakes-jericho_sirens-320k/sp1224-07_having_another.mp3";
//     let _wmut = "E:/Mp3s/We Meet Under Tables/Demos/We Meet Under Tables - Greater Minds.mp3";

//     // let mut decoder = Decoder::new(File::open(&dag_nasty_circles).unwrap());
//     //let buf = std::io::BufReader::new(File::open(&_hotsnakes_having_another).unwrap());
//     let buf = std::io::BufReader::new(File::open(&_dag_nasty_circles).unwrap());
//     let mp3_decoder = Mp3Decoder::new(buf).expect("failed to create mp3 decoder");
//     let source_sample_rate = *(&mp3_decoder.current_frame.sample_rate);
    
//     let stereo_mp3_samples = mp3_decoder.map(|s| s as f64).collect::<Vec<f64>>();
//     let left = stereo_mp3_samples.into_iter().step_by(2).collect();
//     let right = stereo_mp3_samples.into_iter().skip(1).step_by(2).collect();
//     let input_data: Vec<Vec<f64>> = vec![left, right];
//     // let raw_mp3_samples = &mp3_decoder.flat_map(|sample| (sample as f32).to_le_bytes()).collect::<Vec<u8>>();

//     let sample_count = left.len();
//     let track_length_in_seconds = sample_count as f32 / source_sample_rate as f32;

//     println!("\tdevice_sample_rate: {device_sample_rate}");
//     println!("\tsource_sample_rate: {source_sample_rate}");
//     println!("\tsource sample_count: {sample_count}");
//     println!("\ttrack_length_in_seconds: {track_length_in_seconds}");

//     let _resample_total_start = std::time::Instant::now();

//     println!("About to setup resample parameters");

//     let mut input_cursor = std::io::Cursor::new(&raw_mp3_samples);
//     let capacity = (*(&raw_mp3_samples.len()) as f32 * (device_sample_rate as f32 / source_sample_rate as f32)) as usize;
//     let mut output_buffer = Vec::with_capacity(capacity);
//     let mut output_cursor = std::io::Cursor::new(&mut output_buffer);

//     let channels = 2;
//     let chunk_size = 1024; // input data frame size
//     let sub_chunks = 2;

//     println!("About to resample");
//     let resample_start = std::time::Instant::now();
//     ///////////////////////////////////////////////////////
//     // https://github.com/HEnquist/rubato/blob/master/examples/process_f64.rs
//     let nbr_input_frames = input_data[0].len();

//     // Create buffer for storing output
//     let mut outdata = vec![
//         Vec::with_capacity(
//             2 * (nbr_input_frames as f32 * device_sample_rate as f32 / source_sample_rate as f32) as usize
//         );
//         channels
//     ];

//     let f_ratio = device_sample_rate as f64 / source_sample_rate as f64;

//     let mut resampler = rubato::FftFixedInOut::<f64>::new(source_sample_rate as usize, device_sample_rate as usize, 1024, channels).unwrap();
    
//     let mut input_frames_next = resampler.input_frames_next();
//     let resampler_delay = resampler.output_delay();
//     let mut outbuffer = vec![vec![0.0f64; resampler.output_frames_max()]; channels];
//     let mut indata_slices: Vec<&[f64]> = input_data.iter().map(|v| &v[..]).collect();

//     while indata_slices[0].len() >= input_frames_next {
//         let (nbr_in, nbr_out) = resampler
//             .process_into_buffer(&indata_slices, &mut outbuffer, None)
//             .unwrap();
//         for chan in indata_slices.iter_mut() {
//             *chan = &chan[nbr_in..];
//         }
//         append_frames(&mut outdata, &outbuffer, nbr_out);
//         input_frames_next = resampler.input_frames_next();
//     }

//     // Process a partial chunk with the last frames.
//     if !indata_slices[0].is_empty() {
//         let (_nbr_in, nbr_out) = resampler
//             .process_partial_into_buffer(Some(&indata_slices), &mut outbuffer, None)
//             .unwrap();
//         append_frames(&mut outdata, &outbuffer, nbr_out);
//     }

//     let nbr_output_frames = (nbr_input_frames as f32 * device_sample_rate as f32 / source_sample_rate as f32) as usize;
//     println!(
//         "Processed {} input frames into {} output frames",
//         nbr_input_frames, nbr_output_frames
//     );

//     let waves = outdata;
//     let frames_to_skip = resampler_delay;
//     let frames_to_write = nbr_output_frames;
//     let channels = waves.len();

//     // TODO - Write left / right channels back into interleaved buffer.
//     let mut final_audio_buf: Vec<f64> = vec![];
//     let end = frames_to_skip + frames_to_write;
//     // for frame in frames_to_skip..end {
//     //     for wave in waves.iter().take(channels) {
//     //         let value64 = wave[frame];
//     //         let bytes = value64.to_le_bytes();
//     //         final_audio_buf.push(bytes);
//     //     }
//     // }

//     ///////////////////////////////////////////////////////
//     let mut fft_resampler = rubato::FftFixedIn::<f32>::new(
//         source_sample_rate as usize, 
//         device_sample_rate as usize,
//          chunk_size, 
//         sub_chunks,
//          channels
//         ).unwrap();
//     let num_frames_per_channel = fft_resampler.input_frames_next();
//     let sample_byte_size = 8;
//     let num_chunks = &raw_mp3_samples.len() / (sample_byte_size * channels * num_frames_per_channel);

//     println!("num_chunks: {num_chunks}");

//     for _chunk in 0..num_chunks {
//         let waves = read_frames(&mut input_cursor, num_frames_per_channel, channels);
//         let waves_out = fft_resampler.process(&waves, None).unwrap();
//         write_frames(waves_out, &mut output_cursor, channels);
//     }

//     let dur = resample_start.elapsed();

//     println!("Just finished resampling in {:?}ms", dur);

//     let mut audio_buf = output_buffer.into_iter();

//     let mut next_sample = move || {  
//         let byte_one = audio_buf.next().unwrap_or(0u8);
//         let byte_two = audio_buf.next().unwrap_or(0u8);
//         let byte_three = audio_buf.next().unwrap_or(0u8);
//         let byte_four = audio_buf.next().unwrap_or(0u8);

//         let val = f32::from_le_bytes([byte_one, byte_two, byte_three, byte_four]) as i16;
//         // println!("VAL: {val}");
//         val
//     };

//     // // Produce a sinusoid of maximum amplitude.
//     // let mut sample_clock = 0f32;
//     // let mut next_sample = move || {
//     //     sample_clock = (sample_clock + 1.0) % device_sample_rate as f32;
//     //     (sample_clock * 440.0 * 2.0 * std::f32::consts::PI / device_sample_rate as f32).sin()
//     // };

//     let stream = device.build_output_stream(
//         config,
//         move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
//             write_data(data, channels, &mut next_sample)
//         },
//         err_fn,
//         None
//     ).unwrap();

//     stream.play().unwrap();

//     println!("Sleeping");
//     std::thread::sleep(std::time::Duration::from_secs(120));
//     println!("Done!");
// }

// fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> i16) 
// where T: Sample + FromSample<i16>
// {
//     for frame in output.chunks_mut(channels) {
//         let value: T = T::from_sample(next_sample());
//         for sample in frame.iter_mut() {
//             *sample = value;
//         }
//     }
// }

// fn append_frames(buffers: &mut [Vec<f64>], additional: &[Vec<f64>], nbr_frames: usize) {
//     buffers
//         .iter_mut()
//         .zip(additional.iter())
//         .for_each(|(b, a)| b.extend_from_slice(&a[..nbr_frames]));
// }

// // fn play_some_audio() {
// //     // let (sender, receiver) = channel();

// //     let host = cpal::default_host();
// //     let device = host.default_output_device().unwrap();

// //     let mut supported_configs_range = device.supported_output_configs().expect("Error querying output config");
// //     let supported_config = supported_configs_range.next().expect("no output support config!?")
// //         .with_max_sample_rate();
// //     // .with_sample_rate(cpal::SampleRate(44100));

// //     // println!("min sample rate {:?}", supported_config.min_sample_rate());
// //     // println!("max sample rate {:?}", supported_config.max_sample_rate());
    
// //     let output_err_fn = |err| eprintln!("an error occurred on the output audio stream {}", err);
    
// //     let sample_format = supported_config.sample_format();
// //     let config: cpal::StreamConfig = supported_config.into();

    
// //     // play mp3
// //     let _dag_nasty_circles = "E:/Mp3s/20 Years of Dischord/20 Years of Dischord Disk 1/120-20_years_of_dishord-20_dag_nasty-circles.mp3";
// //     let _waiting_room = "E:/Mp3s/Fugazi/Fugazi - 1988 - 13 Songs/01.waiting_room.mp3";
// //     let _eggy_toast = "E:/New Bands/Eggy Toast - Lose your head.mp3.mp3";
// //     let _hotsnakes_having_another = "E:/Mp3s/Hot Snakes/hot_snakes-jericho_sirens-320k/sp1224-07_having_another.mp3";
// //     let _wmut = "E:/Mp3s/We Meet Under Tables/Demos/We Meet Under Tables - Greater Minds.mp3";
    
// //     // let mut decoder = Decoder::new(File::open(&dag_nasty_circles).unwrap());
// //     let buf = std::io::BufReader::new(File::open(&_hotsnakes_having_another).unwrap());
// //     let mut mp3_decoder = Mp3Decoder::new(buf).expect("failed to create mp3 decoder");
// //     // let sample_rate = *(&mp3_decoder.current_frame.sample_rate);
// //     let sample_rate = 44_100;


// //     println!("config sample rate: {}", sample_rate);

// //     let mut all_mp3_samples_f64 = &mp3_decoder.map(|s| s as f32).collect::<Vec<f32>>();    
// //     let sample_count = all_mp3_samples_f64.len() as f32 / 2.0f32;
// //     let track_length_in_seconds = sample_count / sample_rate as f32;

// //     println!("sample_rate: {}, sample_count: {}, track_length_in_seconds: {}", &sample_rate, &sample_count, &track_length_in_seconds);

// //     let resample_total_start = std::time::Instant::now();

// //     let mut left: Vec<f32> = vec![];
// //     let mut right: Vec<f32> = vec![];

// //     for (idx, sample) in all_mp3_samples_f64.iter().enumerate() {
// //         if idx % 2 == 0 {
// //             left.push(*sample);
// //         } else {
// //             right.push(*sample);
// //         }
// //     }

// //     assert!(left.len() == right.len());

// //     println!("About to setup resample parameters");

// //     let params = SincInterpolationParameters {
// //         sinc_len: 256,
// //         f_cutoff: 0.95,
// //         interpolation: SincInterpolationType::Linear,
// //         oversampling_factor: 256,
// //         window: WindowFunction::BlackmanHarris2,
// //     };
    
// //     let mut resampler = SincFixedIn::<f32>::new(
// //         48000 as f64 / 44100 as f64,
// //         2.0,
// //         params,
// //         left.len(),
// //         2
// //     ).unwrap();
    
// //     println!("About to resample");
    
// //     let channels = vec![left, right];

// //     let resample_start = std::time::Instant::now();
// //     let resampled_audio = resampler.process(&channels, None).expect("Failed to resample");
// //     let resampled_start_duration = resample_start.elapsed();

// //     println!("Just finished resampling");


// //     let zip_audio = resampled_audio[0]
// //         .iter()
// //         .zip(&resampled_audio[1])
// //         .collect::<Vec<(&f32, &f32)>>();

// //     let mut vec_channels = vec![];
// //     for z in zip_audio {
// //         vec_channels.push(vec![*z.0, *z.1]);
// //     }

// //     let mut flat_channels = vec_channels
// //         .iter()
// //         .flatten()
// //         .map(|s| *s as i16)
// //         .collect::<Vec<i16>>()
// //         .into_iter()
// //         //.skip((8_864_000 - 1260) * 2);
// //         .skip(0);

// //     let resampled_total_duration = resample_total_start.elapsed();
// //     println!("resampled_total: {:?}ms, resample_process: {:?}ms", resampled_total_duration, resampled_start_duration);


// //     let sample_count_resampled = *(&flat_channels.len()) as f32 / 2.0f32;
// //     let track_length_in_seconds_resampled = sample_count / 48_000 as f32;
// //     println!("sample_rate: {}, sample_count: {}, track_length_in_seconds: {}", 48000, &sample_count_resampled, &track_length_in_seconds_resampled);


// //     let mut next_sample = move || {  
// //         match flat_channels.next() {
// //             Some(sample) => sample,
// //             None => 0i16,
// //         }
// //     };

// //     let output_stream = match sample_format {
// //         SampleFormat::F32 => device.build_output_stream(
// //             &config, 
// //             move |data: &mut [f32], _: &cpal::OutputCallbackInfo| write_sample(data, &mut next_sample),
// //             output_err_fn),
// //         SampleFormat::I16 => device.build_output_stream(
// //             &config, 
// //             move |data: &mut [i16], _: &cpal::OutputCallbackInfo| write_sample(data, &mut next_sample),
// //             output_err_fn),
// //         SampleFormat::U16 => device.build_output_stream(
// //             &config, 
// //             move |data: &mut [u16], _: &cpal::OutputCallbackInfo| write_sample(data, &mut next_sample),
// //         output_err_fn),
// //     }.unwrap();

// //     output_stream.play().unwrap();

// //     println!("Sleeping");
// //     std::thread::sleep(std::time::Duration::from_secs(120));
// //     println!("Done!");
// // }

// // fn write_mp3_frame<T: Sample>(data: &mut [T], next_frame: &mut dyn FnMut() -> Frame) 
// //     where T: cpal::Sample 
// // {
// //     let mp3_frame = next_frame();
// //     let mp3_data = mp3_frame.data;
// //     let _mp3_channels = mp3_frame.channels;
// //     // let sample_rate = mp3_frame.sample_rate;

// //     let mut samples = vec![];
// //     // for incoming_frame in mp3_data.chunks(mp3_channels) {
// //     //     for item in incoming_frame {
// //     //         let value: T = cpal::Sample::from::<i16>(item);
// //     //         samples.push(value);
// //     //     }
// //     // }

// //     for incoming_frame in mp3_data {
// //         let value: T = cpal::Sample::from::<i16>(&incoming_frame);
// //         samples.push(value);
// //     }

// //     for (source, destination) in samples.iter().zip(data.iter_mut()) {
// //         *destination = *source;
// //     }
// // }

// // fn write_sample_f32<T: cpal::Sample>(data: &mut [T], next_sample: &mut dyn FnMut() -> f32)
// // {
// //     for frame in data.chunks_mut(1) {
// //         let value: T = cpal::Sample::from::<f32>(&next_sample());
// //         for sample in frame.iter_mut() {
// //             *sample = value;
// //         }
// //     }        
// // }

// // fn write_sample<T: cpal::Sample>(data: &mut [T], next_sample: &mut dyn FnMut() -> i16)
// // {
// //     for frame in data.chunks_mut(1) {
// //         let value: T = cpal::Sample::from::<i16>(&next_sample());
// //         for sample in frame.iter_mut() {
// //             *sample = value;
// //         }
// //     }        
// // }

// // fn write_sample<T: cpal::Sample>(data: &mut [T], next_sample: &mut dyn FnMut() -> f32)
// // {
// //     for frame in data.chunks_mut(1) {
// //         let value: T = cpal::Sample::from::<f32>(&next_sample());
// //         for sample in frame.iter_mut() {
// //             *sample = value;
// //         }
// //     }        
// // }

// // fn process_incoming_data<T: cpal::Sample>(data: &[T], _channels: usize, tx: &Sender<f32>) 
// // {
// //     for sample in data.iter() {
// //         tx.send(sample.to_f32()).expect("failed sending the incoming stream");
// //     }
// // }

// pub struct Mp3Decoder<R>
// where
//     R: Read + Seek
// {
//     decoder: Decoder<R>,
//     current_frame: Frame,
//     current_frame_offset: usize,
// }


// impl<R> Mp3Decoder<R>
// where
//     R: Read + Seek
// {
//     pub fn new(data: R) -> Result<Self, ()> {
//         let mut decoder = Decoder::new(data);
//         let current_frame = decoder.next_frame().map_err(|_| ())?;

//         Ok(
//             Self {
//                 decoder,
//                 current_frame,
//                 current_frame_offset: 0,
//             }
//         )
//     }
// }

// impl<R> Iterator for Mp3Decoder<R>
// where
//     R: Read + Seek
// {
//     type Item = i16;

//     #[inline]
//     fn next(&mut self) -> Option<i16> {
//         if self.current_frame_offset == self.current_frame.data.len() {
//             match self.decoder.next_frame() {
//                 Ok(frame) => self.current_frame = frame,
//                 _ => return None,
//             }
//             self.current_frame_offset = 0;
//         }

//         let v = self.current_frame.data[self.current_frame_offset];
//         self.current_frame_offset += 1;

//         Some(v)
//     }
// }
