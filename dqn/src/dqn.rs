use anyhow::{anyhow, Context, Result};
use gym_rs::ActionType;
use log::{debug, error, info};
use std::collections::VecDeque;
use std::ops::Mul;
use tch::nn::Module;
use tch::nn::OptimizerConfig;
use tch::{nn, Tensor};

use crate::agent::Stage;
use crate::environment::{
    Action, GameState, GameStatus, LearningResult, Reward, ACTION_SPACE, OBSERVATION_SPACE,
};

// TODO: Make these hyperparameters
const LEARNING_RATE: f64 = 0.01;
const GAMMA: f64 = 0.999;

#[derive(Debug)]
/// A structure representing an action value function Q as defined in the paper.
pub struct ActionValueFunction {
    main_vs: tch::nn::VarStore,
    /// A neural network representing our Q Function
    network: nn::Sequential,
    /// A neural network used in the update process of our Q Function for stability
    target_network: nn::Sequential,
    /// An optimizer for updating our main network
    optimizer: nn::Optimizer<nn::Adam>,
    target_vs: tch::nn::VarStore,
}

impl ActionValueFunction {
    fn create_network(var_store_path: &nn::Path) -> nn::Sequential {
        let input_layer = nn::linear(
            var_store_path / "input_layer",
            OBSERVATION_SPACE,
            24,
            Default::default(),
        );
        let second_layer = nn::linear(var_store_path / "second_layer", 24, 24, Default::default());
        let output_layer = nn::linear(var_store_path / "output_layer", 24, 2, Default::default());

        let network = nn::seq()
            .add(input_layer)
            .add_fn(|xs| xs.relu())
            .add(second_layer)
            .add_fn(|xs| xs.relu())
            .add(output_layer);
        network
    }
    fn new() -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let network = ActionValueFunction::create_network(&vs.root());
        let target_vs = nn::VarStore::new(tch::Device::Cpu);
        let target_network = ActionValueFunction::create_network(&target_vs.root());

        // Stole this from examples
        // https://github.com/LaurentMazare/tch-rs/blob/master/examples/reinforcement-learning/policy_gradient.rs
        let optimizer = tch::nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();

        Self {
            main_vs: vs,
            network,
            optimizer,
            target_network,
            target_vs,
        }
    }

    /// Accepts a batch of experiences and performs a gradient descent update
    /// on the model.
    pub fn learn_from_batch(&mut self, batch: ReplayBatch) -> Result<LearningResult> {
        // Get what the current q network believes to be the value of what it did from memory
        let current_q_eval_of_action_taken = self.network.forward(&batch.state);

        // info!("Q Values of batch of actions:");
        // current_q_eval_of_action_taken.print();

        let batch_actions = batch.actions.to_kind(tch::Kind::Int64).unsqueeze(1);

        // info!("Batch Actions:");
        // batch_actions.print();

        // Extract the values of the actions we actually took
        let current_q_eval_of_action_taken =
            current_q_eval_of_action_taken.gather(1, &batch_actions, false);

        // info!("Gathered:");
        // current_q_eval_of_action_taken.print();p

        // Push that tensor back into a single vector
        let current_q_eval_of_action_taken = current_q_eval_of_action_taken.squeeze();

        // Send the next states through to compute the best value we think we can get
        // if we are not on a terminal state
        let next_q_eval = self.target_network.forward(&batch.next_states);

        // info!("Next State Q:");
        // next_q_eval.print();

        // Get the best - Deleted index
        let max_next_q = next_q_eval.amax(&[1], false);

        // info!("Max next Q:");
        // max_next_q.print();

        // The expected reward is whatever we received for this state + the learning rate times
        // whatever we think we can get if the state is nonterminal.
        let terminal_mask = Tensor::ones(
            &batch.next_state_terminal.size(),
            (tch::Kind::Float, tch::Device::Cpu),
        ) - batch.next_state_terminal;

        // info!("terminal mask");
        // terminal_mask.print();

        let expected_value_of_future = terminal_mask.mul(GAMMA) * max_next_q;
        let expected = batch.rewards.squeeze() + expected_value_of_future;

        // Understanding: Can we look at the weights and see what update was made somehow?
        self.optimizer.zero_grad();

        // info!("Expected Value of future:");
        // expected.print();

        // Understanding: what does the loss actually signify here?
        // TODO: Make loss configurable via hyperparameters
        // Try out using Huber Loss
        // info!("Q:");
        // current_q_eval_of_action_taken.print();
        // info!("Expected:");
        // expected.print();

        let loss = current_q_eval_of_action_taken.huber_loss(&expected, tch::Reduction::None, 1.0);

        // info!("Loss:");
        // loss.print();

        // let loss = current_q_eval_of_action_taken.mse_loss(&expected, tch::Reduction::None);
        // Torch("grad can be implicitly created only for scalar outputs")
        // Use mean_loss instead I guess.

        // // Understanding: Why does taking the mean make sense?
        let mean_loss = loss.mean(tch::Kind::Float);

        mean_loss.backward();
        self.optimizer.step();
        Ok(LearningResult {
            mean_loss: f64::from(mean_loss),
        })
    }

    // Do I need to do some kind of no_grad thing here based on stage?
    pub fn choose_action(&self, state: GameState, _stage: Stage) -> Action {
        // Convert the given game state into a tensor
        // I did this incorrectly... Could I somehow make applying matrices incorrectly a compile time error? Or at least a handled error?
        // Honestly would be kinda lit for shapes of tensors to be encoded in their structure. Do they really need to be variable? Or perhaps some abstract version?
        // People are already looking at this! I'll give it a try in my next project, that would
        // drastically speed up my development time! (Although it has given me a great reason to use rust-gdb
        let observation = Tensor::of_slice(state.state.as_slice())
            .view_(&[1, OBSERVATION_SPACE])
            .to_kind(tch::Kind::Float);
        // Send the tensor through a forward pass of the network
        let output = self.network.forward(&observation);

        // Why do I squeeze? Because there's an extra dimension that doesn't really matter left over. (1,2) -> (2,)
        let choices = output.squeeze();
        let arg_max = choices.argmax(0, false);
        match u8::from(arg_max) {
            a if a <= ACTION_SPACE => ActionType::Discrete(a).into(),
            unknown => {
                error!("Unknown action type {}", unknown);
                // TODO -> Make this Result
                panic!("Unknown action type {}", unknown);
            }
        }
    }

    /// Copies the main to target
    pub fn copy_main_to_target(&mut self) -> Result<()> {
        self.target_vs
            .copy(&self.main_vs)
            .expect("Should be able to copy variable store");
        Ok(())
    }
    /// Saves the main network to a file
    pub fn save_main_network(&self, filename: &str) -> Result<()> {
        self.main_vs.save(filename)?;
        Ok(())
    }
}

// type MemoryNode = (GameState, Action, Reward, GameState);

pub struct ReplayMemory {
    sampling_size: u8,
    storage: ReplayStorage,
}

/// A grouping of experiences used for both storage and updates to the
/// net
pub struct ReplayStorage {
    pub states: VecDeque<GameState>,
    pub actions: VecDeque<Action>,
    pub rewards: VecDeque<Reward>,
    pub next_states: VecDeque<GameState>,
}

/// A Batch of experiences stored as tensors for easy updates
pub struct ReplayBatch {
    pub state: Tensor,
    pub actions: Tensor,
    pub rewards: Tensor,
    pub next_states: Tensor,
    pub next_state_terminal: Tensor,
}

impl ReplayMemory {
    pub fn store(
        &mut self,
        previous_state: GameState,
        action_taken: Action,
        reward: Reward,
        next_state: GameState,
    ) -> Result<()> {
        self.storage.states.push_back(previous_state);
        self.storage.actions.push_back(action_taken);
        self.storage.rewards.push_back(reward);
        self.storage.next_states.push_back(next_state);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.storage.states.len()
    }

    /// Returns a ReplayBatch, or None if not enough samples have been stored
    pub fn sample(&self) -> Result<ReplayBatch> {
        let indices = rand::seq::index::sample(
            &mut rand::thread_rng(),
            self.storage.states.len(),
            self.sampling_size.into(),
        );

        let (mut states, mut actions, mut rewards, mut next_states) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());

        // Create a random sample minibatch
        // This is safe because of the way we sample and store into the experience
        // replay buffer
        for index in indices {
            states.push(self.storage.states[index].clone());
            actions.push(self.storage.actions[index].clone());
            rewards.push(self.storage.rewards[index].clone());
            next_states.push(self.storage.next_states[index].clone());
        }

        // Convert the vectors of floats into tensors
        let prev_observations: Vec<Tensor> = states
            .into_iter()
            .map(|state| {
                Tensor::of_slice(state.state.as_slice())
                    // .view_(&1, OBSERVATION_SPACE)
                    .to_kind(tch::Kind::Float)
            })
            .collect();

        // Convert the state tensors into one large tensor for batch processing
        let batch_observations = Tensor::stack(prev_observations.as_slice(), 0);

        // Convert the actions_taken into a tensor for batch processing
        // We make the assumption here that actions are discrete, and throw an error otherwise
        let discrete_actions: Result<Vec<u8>> = actions
            .into_iter()
            .map(|action| match action.0 {
                ActionType::Discrete(value) => Ok(value),
                _ => Err(anyhow!(
                    "Found continuous action when assuming actions were discrete",
                )),
            })
            .collect();
        let discrete_actions = discrete_actions.context("learning dqn")?;
        let batch_actions = Tensor::of_slice(discrete_actions.as_slice()).to_kind(tch::Kind::Uint8);
        let next_state_terminal: Vec<f64> = next_states
            .iter()
            .map(|state| match state.status {
                GameStatus::Terminal => 1.0,
                GameStatus::Ongoing => 0.0,
            })
            .collect();

        let next_state_terminal =
            Tensor::of_slice(next_state_terminal.as_slice()).to_kind(tch::Kind::Float);

        let rewards = Tensor::of_slice(
            rewards
                .into_iter()
                .map(|reward| reward.0)
                .collect::<Vec<f64>>()
                .as_slice(),
        )
        .to_kind(tch::Kind::Float);

        let next_states: Vec<Tensor> = next_states
            .into_iter()
            .map(|state| Tensor::of_slice(state.state.as_slice()).to_kind(tch::Kind::Float))
            .collect();
        let next_states = Tensor::stack(next_states.as_slice(), 0);

        Ok(ReplayBatch {
            state: batch_observations,
            actions: batch_actions,
            rewards,
            next_states,
            next_state_terminal,
        })
    }

    pub fn new(sampling_size: u8) -> ReplayMemory {
        ReplayMemory {
            storage: ReplayStorage {
                states: VecDeque::new(),
                actions: VecDeque::new(),
                rewards: VecDeque::new(),
                next_states: VecDeque::new(),
            },
            sampling_size,
        }
    }
}

/// A structure implementing the DQN algorithm using experience replay
pub struct DQNAgent<'a> {
    /// Replay memory of sequence and action taken to result
    pub replay_memory: &'a mut ReplayMemory,
    /// The model the agent will use
    pub model: ActionValueFunction,
}

impl<'a> DQNAgent<'a> {
    pub fn new(replay_memory: &'a mut ReplayMemory) -> DQNAgent<'a> {
        Self {
            replay_memory,
            model: ActionValueFunction::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        let _ = env_logger::builder()
            .filter(None, LevelFilter::Info)
            .is_test(true)
            .try_init();
    }

    #[test]
    fn test_network_shapes() {
        init();
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let network = ActionValueFunction::new(&vs);

        let initial_game_state = vec![1.0, 1.0, 1.0, 1.0];
        let action = network.choose_action(initial_game_state);
        match action.0 {
            ActionType::Discrete(action) => {
                info!("Correctly chose a discrete action!");
            }
            _ => panic!("Unexpected action type"),
        }
    }
}
