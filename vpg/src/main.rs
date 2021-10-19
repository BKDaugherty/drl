use crate::agent::{Agent, Stage};
use crate::args::{PerformArgs, TrainArgs};
use crate::environment::{run_episode, Action, EpisodeResult, LearningResult};
use anyhow::{Context, Result};
use env_logger::Env;
use gym_rs::{ActionType, GifRender, GymEnv, MountainCarEnv};
use log::info;
use plotters::prelude::*;
use structopt::StructOpt;
use tch::nn;

mod agent;
mod args;
mod environment;

#[derive(Debug, StructOpt)]
struct Args {
    #[structopt(subcommand)]
    cmd: Command,
}
#[derive(Debug, StructOpt)]
enum Command {
    Train(TrainArgs<HyperParameters>),
    Perform(PerformArgs),
}

#[derive(Debug, StructOpt)]
struct HyperParameters {}

fn main() {}
