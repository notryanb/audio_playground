//use crate::audio_ex::run_example;
use crate::synth::run;

//mod audio_ex;
mod synth;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    run().unwrap();
    println!("Done");
    Ok(())
}
