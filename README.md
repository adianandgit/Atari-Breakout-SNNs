# Neuromorphic Deep RL: Atari Breakout via SNNs & Rainbow DQN

This project explores the integration of Spiking Neural Networks (SNNs) with advanced Deep Reinforcement Learning (RL). It implements a robust Deep Q-Network (DQN) agent to master the Atari Breakout environment, and subsequently converts this trained Artificial Neural Network (ANN) into an energy-efficient Spiking Neural Network (SNN).

By bridging state-of-the-art RL techniques with neuromorphic principles, this repository studies performance trade-offs, computational efficiency, and the robustness of replacing dense matrix operations with sparse, event-driven spike communications.

---

## 🚀 Algorithm Highlights

This agent builds upon Vanilla DQN by incorporating several components of the "Rainbow" DQN architecture, alongside a complete neuromorphic conversion pipeline.

### Deep Reinforcement Learning (ANN)

* 
**Double DQN (DDQN)**: Reduces the overestimation bias inherent in standard Q-learning by decoupling the action selection and evaluation steps.


* 
**Prioritized Experience Replay (PER)**: Improves sample efficiency by sampling experiences with non-uniform probability, heavily weighing transitions with high Temporal-Difference (TD) errors.


* 
**N-Step Returns**: Spreads the states influencing the rewards across the timeline, significantly improving credit assignment in sparse-reward environments.


* 
**Noisy Networks**: Replaces traditional $\epsilon$-greedy strategies by injecting learned parameter noise into the network's output layers for dynamic exploration.


* 
**Dueling Networks**: Decomposes Q-values into a state value function and an advantage function, improving learning stability when many actions are equally valuable.



### Spiking Neural Networks (SNN Integration)

* 
**Neuromorphic Efficiency**: Simulates the brain's event-driven processing, activating computation only when spikes occur, which drastically reduces theoretical energy usage.


* 
**Leaky Integrate-and-Fire (LIF) Neurons**: Replaces standard ReLU activations with biologically inspired neurons that accumulate membrane potential over time and fire discrete spikes upon crossing a threshold.


* 
**Poisson Rate Coding**: Encodes continuous input frames (pixels) into sparse spike trains, where the probability of a spike is proportional to the pixel intensity.


* 
**ANN-to-SNN Conversion**: Utilizes a robust direct weight-transfer methodology, bypassing the instability of Surrogate Gradients and Spike-Timing Dependent Plasticity (STDP) in deep RL environments.



---

## ⚙️ Mathematical Foundations

### Reinforcement Learning

The fundamental Q-learning update step iteratively reduces the Temporal Difference (TD) error using the Bellman Equation:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

To better propagate delayed rewards, we implement N-step returns:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n V(s_{t+n})$$

### LIF Neuron Dynamics

In the SNN, the membrane potential $v(t)$ of each LIF neuron is governed by the following differential equation:

$$\tau \frac{dv(t)}{dt} = -(v(t) - v_{rest}) + \sum_{i} W_i Input_i$$

When $v(t) \ge v_{thresh}$, the neuron emits a binary spike and the potential resets.

---

## 📂 Project Structure

```text
.
├── Agent_Implementation.py                 # Core agent logic and actions
├── Atari_Environment.py                    # Setup and wrapping of the OpenAI Gym environment
├── config.py                               # Hyperparameter definitions
├── DQN_Architecture.py                     # Standard and Double DQN architectures
├── Image_Processing_and_Frame_Stacking.py  # Visual preprocessing for 84x84 stacked frames
├── Main.py                                 # Main training loop for ANN
├── NOISY_Dueling_DQN.py                    # Advanced network with noise and dueling streams
├── PER.py                                  # Prioritized Experience Replay buffer
├── Test.py                                 # Evaluation and rendering of the trained model
├── training_function.py                    # Optimization and loss calculation steps
├── Visualisation_and_plotting.py           # Generation of training metric graphs
└── Playing-Atari-with-SNNs-main/           # Dedicated directory for SNN conversion
    ├── Agent.py
    ├── Architectures.py
    ├── Environment.py
    ├── Neuron.py                           # Implementation of LIF Neurons
    ├── Train.py
    ├── Test.py
    └── weights/                            # Stores binary and greyscale model weights

```

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone "https://github.com/adianandgit/Atari-Breakout-SNNs.git"

```

### 2. Install Dependencies

Create a separate virtual environment, activate it, and run:

```bash
pip install -r requirements.txt

```

### 3. Run the Pre-trained Model

```bash
python Test.py

```

---

## 📊 Configuration

Edit `config.py` to tune the hyperparameters for the DQN training phase:

```python
config = {
    "gamma": 0.99,
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "batch_size": 32,
    "alpha": 0.6,
    "beta_start": 0.4,
    "n_step": 3,
    "target_update_freq": 1000,
    "initial_replay_size": 5000,
}

```

---

## 📈 Training Methods and Results

### ANN Training & Fine-Tuning

* The model was initially trained using a Double-Deep Q Network for 2000 episodes, resulting in a trailing 50-episode average score of **81.60**.
* N-step Learning was subsequently introduced. Instead of training from scratch, the neural network was initialized with the previous model's weights to provide a strong starting baseline.
* After 1500 additional episodes of fine-tuning, the model reached an average score of **169** over the final 50 training episodes, with a peak score of **290** (far outperforming the human baseline of 32).



To fine-tune the existing agent:

```bash
python Main.py

```

*(Note: To train entirely from scratch, remove lines 12 and 13 from `Main.py` prior to running the above command).*

### ANN-to-SNN Conversion Methodology

Because direct SNN training using Surrogate Gradients proved highly unstable and computationally expensive for deep RL (e.g., 25 steps taking ~15 minutes) , this project utilizes a direct **Weight Transfer** approach:

1. 
**Transfer:** Weights from the trained ANN are directly copied into the SNN architecture.


2. 
**Scaling:** To compensate for reduced spiking activity in deeper convolutional layers, layer-wise scaling is applied (e.g., Layer 1 scaled by 20x, Layer 2 by 100x).


3. 
**Inference:** The network interprets Poisson-encoded input frames, and the action layer neuron with the highest spike count over a simulation window is selected as the output.



**SNN Results:** The converted SNN successfully avoided complex gradient computations, acting as an efficient real-time simulation model. Evaluation histograms indicate that the SNN achieved a mean reward distribution of **1.13**, compared to the baseline ANN's **0.63** during the same testing timeframe.

---

## 🛠️ Using Noisy Nets and Dueling Networks

Noisy Nets and Dueling networks can further enhance model exploration and state-value estimation. To use Noisy Nets with the Double DQN, you need to modify the action selection logic.

Replace lines 70-85 in `Agent_Implementation.py` with:

```python
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.to(self.policy_net.device)
        # Get Q-values from the policy network and select action with max Q-value.
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

```

Next, add the noise reset function inside your environment stepping/loop around line 130 in `Agent_Implementation.py`:

```python
        policy_net.reset_noise()

```

---

## 👥 Acknowledgements

This architecture heavily relies on foundational research in the Deep RL and Neuromorphic spaces:

* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461)
* [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952)
* Reinforcement Learning: An Introduction (Sutton and Barto) for N-step Learning
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581)
* [Noisy Networks for Exploration](https://arxiv.org/pdf/1706.10295)

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
