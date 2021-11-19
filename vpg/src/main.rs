use crate::agent::{Agent, Stage};
use crate::args::{PerformArgs, TrainArgs};
use crate::environment::{run_episode, Action, EpisodeResult, LearningResult};
use anyhow::{Context, Result};
use env_logger::Env;
use gym_rs::{ActionType, CartPoleEnv, GifRender, GymEnv, MountainCarEnv};
use log::info;
use plotters::prelude::*;
use structopt::{clap::arg_enum, StructOpt};
use tch::nn::{Module, OptimizerConfig};
use tch::{nn, Tensor};
use tch_distr::{Categorical, Distribution};

mod agent;
mod args;
mod environment;

arg_enum! {
    #[derive(Debug, Clone, StructOpt)]
    pub enum RewardCalculation {
    All,
    RewardToGo,
    }
}

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

#[derive(Debug, StructOpt, Clone)]
struct HyperParameters {
    #[structopt(long, default_value = "5000")]
    batch_size: usize,
    #[structopt(long, default_value = "0.01")]
    learning_rate: f64,
    #[structopt(long, default_value = "All")]
    reward_calculation: RewardCalculation,
}

struct VPG {
    vs: nn::VarStore,
    network: nn::Sequential,
    pub optimizer: nn::Optimizer<nn::Adam>,
}

impl VPG {
    fn new(
        observation_space: u32,
        hidden_layer_size: u32,
        action_space: u32,
        learning_rate: f64,
    ) -> Self {
        // Create VPG network
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let vs_path = &vs.root();
        let input_layer = nn::linear(
            vs_path / "input_layer",
            observation_space.into(),
            hidden_layer_size.into(),
            Default::default(),
        );
        let output_layer = nn::linear(
            vs_path / "second_layer",
            hidden_layer_size.into(),
            action_space.into(),
            Default::default(),
        );
        let network = nn::seq()
            .add(input_layer)
            .add_fn(|xs| xs.tanh())
            .add(output_layer);

        let optimizer = tch::nn::Adam::default().build(&vs, learning_rate).unwrap();

        Self {
            vs,
            network,
            optimizer,
        }
    }
    fn get_policy(&self, observation: Tensor) -> Result<Categorical> {
        let logits = self.network.forward(&observation);
        Ok(Categorical::from_logits(logits))
    }
    fn get_action(&self, observation: Tensor) -> Result<Action> {
        let dist = self
            .get_policy(observation)
            .context("Getting distribution for action")?;
        // Uhhh...

        Ok(Action(ActionType::Discrete(u8::from(dist.sample(&[1])))))
    }
    fn compute_loss(&self, obs: Tensor, actions: Tensor, weights: Tensor) -> Result<Tensor> {
        let log_probabilities = self.get_policy(obs)?.log_prob(&actions);
        Ok(-(log_probabilities * weights).mean(tch::Kind::Float))
    }

    fn rewards_to_go(rewards: Vec<f64>) -> Vec<f64> {
        let mut rewards_to_go = Vec::new();
        let mut total_reward = 0.0;
        for reward in rewards.iter().rev() {
            total_reward += reward;
            rewards_to_go.push(total_reward);
        }
        rewards_to_go.into_iter().rev().collect()
    }
}

fn train_agent_one_epoch(
    hp: HyperParameters,
    env: &mut dyn GymEnv,
    vpg: &mut VPG,
    render: &mut Option<GifRender>,
) -> Result<Vec<f64>> {
    // Initialize batch carriers
    let (mut batch_obs, mut batch_acts, mut batch_weights, mut batch_rets, mut batch_lens) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut episode_rewards = Vec::new();
    let mut obs = env.reset();

    let mut done_rendering_first_episode = false;

    // Gather experience using the current policy
    loop {
        batch_obs.push(obs.clone());

        // TODO: Convert observation into Tensor
        let action = vpg.get_action(Tensor::of_slice(obs.as_slice()).to_kind(tch::Kind::Float))?;
        let (next_state, reward, game_complete, info) = env.step(action.clone().0);

        obs = next_state;
        batch_acts.push(action.discrete_value()?);
        episode_rewards.push(reward);

        match (done_rendering_first_episode, render.as_mut()) {
            (false, Some(renderer)) => env.render(renderer),
            _ => {}
        }

        if game_complete {
            let episode_length = episode_rewards.len();
            let episode_return: f64 = episode_rewards.iter().cloned().sum();
            batch_rets.push(episode_return);
            batch_lens.push(episode_length);

            if !done_rendering_first_episode {
                done_rendering_first_episode = true;
            }

            // Set weights for each (state, action) pair to be episode_return
            match hp.reward_calculation {
                RewardCalculation::All => {
                    for _ in 0..episode_length {
                        batch_weights.push(episode_return);
                    }
                }
                RewardCalculation::RewardToGo => {
                    batch_weights = VPG::rewards_to_go(episode_rewards);
                }
            }

            obs = env.reset();
            episode_rewards = Vec::new();

            // If we've gathered enough experience this episode, we are done
            if batch_obs.len() > hp.batch_size {
                break;
            }
        }
    }

    let tensor_observations: Vec<Tensor> = batch_obs
        .into_iter()
        .map(|obs| Tensor::of_slice(obs.as_slice()).to_kind(tch::Kind::Float))
        .collect();

    let tensor_observations = Tensor::stack(tensor_observations.as_slice(), 0);

    // Use the experience gathered to optimize our network
    vpg.optimizer.zero_grad();
    let batch_loss = vpg
        .compute_loss(
            tensor_observations,
            Tensor::of_slice(batch_acts.as_slice()).to_kind(tch::Kind::Uint8),
            Tensor::of_slice(batch_weights.as_slice()).to_kind(tch::Kind::Float),
        )
        .context("Computing loss")?;

    batch_loss.backward();
    vpg.optimizer.step();

    Ok(batch_rets)
}

fn main() -> Result<()> {
    // Create our logger
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let args = Args::from_args();
    let mut env = CartPoleEnv::default();

    match args.cmd {
        Command::Train(train_args) => {
            let mut vpg = VPG::new(4, 32, 2, train_args.hyperparameters.learning_rate);

            for epoch in 0..train_args.epochs {
                // Create a new renderer
                let render = GifRender::new(540, 540, &format!("img/basic_vpg_{}.gif", epoch), 20)
                    .expect("should be able to create renderer");

                let episode_rewards = train_agent_one_epoch(
                    train_args.hyperparameters.clone(),
                    &mut env,
                    &mut vpg,
                    &mut Some(render),
                )
                .context("Training epoch")?;
                let episode_count = episode_rewards.len();
                let average_reward: f64 =
                    episode_rewards.iter().cloned().sum::<f64>() / episode_count as f64;
                let max_reward = episode_rewards
                    .iter()
                    .cloned()
                    .reduce(f64::max)
                    .expect("Reward shouldn't be empty");
                let min_reward = episode_rewards
                    .iter()
                    .cloned()
                    .reduce(f64::min)
                    .expect("Reward shouldn't be empty");
                info!(
                    "Epoch {} complete after {} episodes. Rewards -> Avg: {}, Max: {}, Min: {}",
                    epoch, episode_count, average_reward, max_reward, min_reward
                );
            }
        }
        Command::Perform(perform_args) => {}
    }
    Ok(())
}
