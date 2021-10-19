use anyhow::{Context, Result};
use env_logger::Env;
use gym_rs::{ActionType, CartPoleEnv, GifRender, GymEnv};
use log::info;
use rand::Rng;
use std::path::PathBuf;
use std::time::SystemTime;
use structopt::StructOpt;
use tch::nn;

// TODO: Figure out only what I need.
use plotters::prelude::*;

pub mod agent;
pub mod dqn;
pub mod environment;

use crate::agent::{Agent, Stage};
use crate::dqn::{DQNAgent, ReplayMemory};
use crate::environment::{Action, EpisodeResult, GameState, Reward};

#[derive(Debug, StructOpt)]
struct Args {
    #[structopt(subcommand)]
    cmd: Command,
}
#[derive(Debug, StructOpt)]
enum Command {
    Train(TrainArgs),
    Perform(PerformArgs),
}

#[derive(Debug, StructOpt)]
struct HyperParameters {
    #[structopt(long, default_value = "0.71")]
    learning_rate: f64,
    // Doesn't currently do anything...
    #[structopt(long, default_value = "0.6")]
    discount_factor: f64,
    #[structopt(long, default_value = "32")]
    minibatch_size: u8,
    #[structopt(long, default_value = "64")]
    replay_memory_min_size: usize,
    #[structopt(long, default_value = "10000")]
    num_step_copy_target: u32,
    #[structopt(long, default_value = "4")]
    num_step_train: u32,
    #[structopt(long, default_value = "50000")]
    replay_memory_max_size: u32,
    #[structopt(long, default_value = "0.01")]
    epsilon_decay: f64,
    #[structopt(long, default_value = "0.01")]
    min_epsilon: f64,
    #[structopt(long, default_value = "1.0")]
    max_epsilon: f64,
}

#[derive(Debug, StructOpt)]
struct PerformArgs {
    /// The path from which to load the neural network
    #[structopt(long)]
    load_path: PathBuf,
}

#[derive(Debug, StructOpt)]
struct TrainArgs {
    /// The path to which to save the trained neural network
    #[structopt(long, default_value = "./networks")]
    save_path: PathBuf,
    #[structopt(long, default_value = "5000")]
    episodes: u32,
    /// Number of training epochs to run
    #[structopt(long, default_value = "5")]
    epochs: u32,
    /// Whether or not we should plot results
    #[structopt(long)]
    plot_results: bool,
    #[structopt(flatten)]
    hyperparameters: HyperParameters,
    /// Whether or not networks should be saved for later use
    #[structopt(long)]
    save_networks: bool,
}

impl Agent for DQNAgent<'_> {
    fn choose_action(&mut self, state: GameState) -> Action {
        self.model.choose_action(state, Stage::Test)
    }
}

fn run_episode(
    agent: &mut dyn Agent,
    env: &mut dyn GymEnv,
    render: &mut Option<GifRender>,
) -> Result<EpisodeResult> {
    let mut total_reward = 0.0;
    // Reset the environment before running through an episode
    let mut game_state = GameState::new(env.reset(), false);

    // Run the agent on the environment
    loop {
        let action = agent.choose_action(game_state);

        // Run a step of the environment
        let (next_state, reward, game_complete, info) = env.step(action.0);

        total_reward += reward;

        if let Some(info) = info {
            info!("Step information: {}", info);
        }
        // Advance the game state
        let next_state = GameState::new(next_state, game_complete);
        game_state = next_state;

        // Render the frame
        if let Some(render) = render.as_mut() {
            env.render(render);
        }

        // Break if the game is complete
        if game_complete {
            break;
        }
    }

    Ok(EpisodeResult {
        total_reward,
        learning_summaries: Vec::new(),
    })
}

fn train_dqn(dqn: &mut DQNAgent, train_args: &TrainArgs) -> Result<Vec<EpisodeResult>> {
    let mut episode_results = Vec::new();
    // Counter for step, used to count when we should train
    let mut step = 0;
    // Epsilon used to train using Epsilon Greedy Exploration
    let mut epsilon = 1.0;

    for episode in 0..train_args.episodes {
        // Create the cart pole environment
        let mut env = CartPoleEnv::default();

        let mut episode_result = EpisodeResult::default();
        let mut episode_reward = 0.0;
        let mut game_state = GameState::new(env.reset(), false);
        loop {
            // Depending on epsilon, use the model to choose an action, or choose randomly
            let mut rng = rand::thread_rng();
            let random = rng.gen_range(0.0..1.0);

            let action = if random > epsilon {
                // Run a forward pass to get the best action
                dqn.model.choose_action(game_state.clone(), Stage::Train)
            } else {
                Action(ActionType::Discrete(rng.gen_range(0..2)))
            };

            // emulate that action in the environment
            let (next_state, reward, game_complete, info) = env.step(action.clone().0);
            let next_state = GameState::new(next_state, game_complete);

            episode_reward += reward;

            // Store this transition into the replay memory
            dqn.replay_memory
                .store(
                    game_state.clone(),
                    action.clone(),
                    Reward(reward),
                    next_state.clone(),
                )
                .context("storing memory node")?;

            // Based on step, update model parameters
            if step % train_args.hyperparameters.num_step_copy_target == 0 {
                dqn.model.copy_main_to_target()?;
            }

            if step % train_args.hyperparameters.num_step_train == 0 {
                // Update gradients based on a bunch of samples, not just this one to avoid
                // locality problems
                if dqn.replay_memory.len() > train_args.hyperparameters.replay_memory_min_size {
                    let batch = dqn
                        .replay_memory
                        .sample()
                        .context("Sampling for a minibatch of updates")?;
                    let summary = dqn
                        .model
                        .learn_from_batch(batch)
                        .context("Learning from sampled batch in episode")?;
                    // info!("Loss: {}", summary.mean_loss);
                    episode_result.learning_summaries.push(summary);
                }
            }

            game_state = next_state.clone();

            if let Some(info) = info {
                info!("Step information: {}", info);
            }

            // Update step
            step += 1;

            if game_complete {
                break;
            }
        }
        let hyper = &train_args.hyperparameters;
        // Update epsilon so we explore less next time
        epsilon = hyper.min_epsilon
            + (hyper.max_epsilon - hyper.min_epsilon)
                * ((-hyper.epsilon_decay * episode as f64).exp());
        // info!("Epsilon: {}", epsilon);
        episode_result.total_reward = episode_reward;
        episode_results.push(episode_result);
    }
    Ok(episode_results)
}

fn plot_results(path: &str, graph_title: &str, episode_results: Vec<EpisodeResult>) -> Result<()> {
    let num_episodes = episode_results.len();

    let episode_rewards: Vec<f64> = episode_results
        .iter()
        .map(|episode_result| episode_result.total_reward)
        .collect();

    let losses: Vec<f64> = episode_results
        .into_iter()
        .map(|e| e.learning_summaries)
        .flatten()
        .map(|lr| lr.mean_loss)
        .collect();

    let loss_filename = format!("{}{}-loss.png", path, graph_title);
    let loss_root = BitMapBackend::new(&loss_filename, (640, 480)).into_drawing_area();
    loss_root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&loss_root)
        .caption(
            format!("{} - Loss over Episode", graph_title),
            ("sans-serif", 50).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0usize..losses.len() as usize, -0.0f64..10.0f64)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(losses.into_iter().enumerate(), &BLUE))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    let reward_filename = format!("{}{}-reward.png", path, graph_title);
    let reward_root = BitMapBackend::new(&reward_filename, (640, 480)).into_drawing_area();
    reward_root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&reward_root)
        .caption(
            format!("{} - Reward over Episode", graph_title),
            ("sans-serif", 50).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0usize..num_episodes, 0.0f64..250f64)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            episode_rewards.into_iter().enumerate(),
            &BLUE,
        ))?
        .label("Reward")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}

fn main() -> Result<()> {
    // Create our logger
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let args = Args::from_args();
    match args.cmd {
        Command::Train(train_args) => {
            for epoch in 0..train_args.epochs {
                let mut replay_memory =
                    ReplayMemory::new(train_args.hyperparameters.minibatch_size);
                let mut agent = DQNAgent::new(&mut replay_memory);
                let episode_results = train_dqn(&mut agent, &train_args)?;
                if train_args.plot_results {
                    plot_results("./plots/", &format!("{}", epoch), episode_results)?;
                } else {
                    let avg_reward = episode_results
                        .iter()
                        .fold(0.0, |acc, x| acc + x.total_reward)
                        / train_args.episodes as f64;
                    info!("Average Reward for epoch {}: {}", epoch, avg_reward);
                }

                if train_args.save_networks {
                    agent.model.save_main_network(&format!(
                        "{}/{}",
                        train_args.save_path.to_str().expect("save path is invalid"),
                        epoch,
                    ))?;
                }
            }
        }
        Command::Perform(perform_args) => {
            let mut env = CartPoleEnv::default();
            // I shouldn't need this...
            // Replay Memory is only needed for training...
            let mut replay_memory = ReplayMemory::new(1);
            let mut vs = nn::VarStore::new(tch::Device::Cpu);
            let mut agent = DQNAgent::new(&mut replay_memory);
            vs.load(
                perform_args
                    .load_path
                    .to_str()
                    .expect("Load path is invalid"),
            )
            .context("Loading network")?;
            // Get a timestamp to mark our results
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("time went backwards")
                .as_secs();

            let render = GifRender::new(540, 540, &format!("img/cart_pole_{}.gif", now), 20)
                .expect("Should be able to create gif");

            let result = run_episode(&mut agent, &mut env, &mut Some(render))
                .context("Running test episode")?;

            info!(
                "Model finished the game with a total reward of {}",
                result.total_reward
            );
        }
    }

    Ok(())
}
