# Vanilla DQN in Rust

This is my attempt at an implementation of a Vanilla DQN as described [here](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), but in Rust!

## Why Rust?

I don't really like writing Python code, and find the developer time to take too long, so I decided I'm going to fuel my ML journey with a type-safe langauge. Maybe I'll write bindings if someone makes me write Python, but I'd really just prefer not to :P

## Why this algorithm?

We read it in my paper club, and I wanted to implement it!

## Installation

This depends on [tch-rs](https://github.com/LaurentMazare/tch-rs), so follow instructions there first.

Specifically you'll want to run `cargo build -j 1` once you've done the above.

## Anything below this is probably not all that interesting, and is just where I dump my thoughts
- Need to see how to train a Pytorch algo in general
- Check this tutorial out: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

- Maybe this one is best for computing loss: https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb
- Read through this one too: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

- I made the sequences way too complicated. I don't fully understand why 
 "Set st+1 = st,at,xt+1 and preprocess φt+1 = φ(st+1)"
 
 int this statement, st+1 has to include at. It doesn't look like it's ever used, and only the frames themselves are used 
 
Check out [this answer](https://ai.stackexchange.com/questions/21692/how-to-convert-sequences-of-images-into-state-in-dqn).

All most all of my errors have been because of bad shapes with Tensors (admittedly I have no idea what I'm doing) but it's a little sad that all of these could be compiler errors.

[This seems super cool](https://github.com/jerry73204/rust-typed-tensor) as does [this discussion](https://internals.rust-lang.org/t/tensors-static-typing/13114/19). I think I'll try using this out the next time I work with Tensors.


Spent a lot of time trying to get a debugger to work which would really improve my coding speed, and would be helpful in general!

Finally got my debugger to work by just setting LD_LIBRARY_PATH manually
find the path on your system via:

`find / -regex ".*libtorch_cpu.so.*" 2>/dev/null`
`LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/brendon/Dev/drl/dqn/target/debug/build/torch-sys-ef5a657cb01ae5c2/out/libtorch/libtorch/lib" rust-gdb ./target/debug/dqn`

Got this error which is beat :(
 -> Torch("grad can be implicitly created only for scalar outputs")
 
Looks like for the C Library I need to explicitly register my gradients? Will need to see how this
is done... But good work! So close I think!

loss of each sample, and every single element in loss has been averaged on batch, should I use loss.mean().backward()?


### Helpful torch and other things
*.forward*: Run a tensor through the network
*.gather*: Gathers the values along a specific axis

Rust unzip is good for Vec<(T, U)> -> (Vec<T>, Vec<U>), but doesn't work for variable length :(
- Types :(

#### Doing some Debugging

I seem to only be taking the action push to the left ever...

Things I need to know
- What is a reasonable Q Value for an action?
- I take the max of the two Q Values right? That's how I decide what action to take? Can we compare the two?

Maybe I'm just not training enough? Without epochs, I don't ever give a new random start to the algorithm... I wonder if I'm calculating rewards correctly though?

Let's try epochs..

Replay Memory should be kept out of DQN. Perhaps it just gets a reference to the ReplayMemory instead.

Maybe start over and try using this: [DQN tutorial](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc)
- Things I haven't done: Target vs Main Network
- Exploration Strategy
- Learning every 4 steps as opposed to every step
- Smart initialization of weights
  - Should I try the He Uniform Initalization of weights?
- exploring different loss functions!
- Optimizing replay buffer to not be slow as shit (Max Size, maybe not 4 separate arrays?)
  - Use VecDeque for specific size
- Running in release mode

Implementation Guides
- How do I copy values of my NN from Main to Target
- Steps should be exposed over an epoch in order to better train the network
  - Copy to Target every 100
  - Train on a batch every 4
  - Epsilon should be a function of episodes or maybe steps?
- Huber Loss instead of MSE
- Hyperparameters
  - Min Replay Size (1000)
  - Batch Size
  - Replay Total Size (50000)
  - Learning Rate (?)
  - Discount Factor (?)
    - Used for max future q
  - Epsilon (They edit epsilon every episode)
    - Epsilon Update: epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

Second

- Epoch has Steps and Episodes
  (Steps are important meta information)
  - Epsilon should be passed in to the network so it can decide

- Devlog: Currently implementing the main target value swap. Messing with DQN AV function.
- Need a good way to clone my sequential network. -> do so by coupling main_vs and network then impl clone

Okay I implemented epsilon, and target network and parameterized a bunch of things.
- Still need to do...
  - Learning Rate
  - Check my math -> Seems I am learning to fail? (Increasing episodes makes me lose consistently...)
  - Should Replay Memory be shared over epochs? I think no?
  
- Also look into how I can evaluate / log interesting things if I want to!

Ok so I did some plotting, and although I'm getting interesting results, I think maybe I'm doing something wildly incorrect, or maybe drl really just is quite brittle. Time to try implementing something simpler first. I'll be back DQN!
