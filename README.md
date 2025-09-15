# Estimation Bias in Reinforcement Learning

An empirical study of estimation bias in continuous control reinforcement learning algorithms, investigating bias correction mechanisms and their impact on learning performance. Developed during a summer 2025 internship at ISIR (Institut des Systèmes Intelligents et de Robotique), Sorbonne University, Master AI2D M1.

## About Estimation Bias in RL

Reinforcement learning agents learn optimal decision-making by interacting with environments. In modern applications, these agents use neural networks to estimate action values, but these estimations contain errors called estimation bias. These errors can propagate and amplify during learning through bootstrapping, degrading action selection quality.

A particular problem occurs when agents must select the highest-valued action: they preferentially choose overestimated actions due to noise, creating a cycle that amplifies errors known as maximization bias. Current state-of-the-art methods address this through pessimistic approaches that deliberately underestimate values to counteract overestimation.

## Usage

```sh
# Setup environment
nix develop

# Train algorithms on MountainCar
just train-mountaincar

# Compute Monte Carlo bias measurements
just compute-monte-carlo --env mountaincar --algo sac

# Reproduce TQC figure analysis
just compute-tqc-figure

# Generate experimental reports
just paper-compile
```

## Project Structure

```
src/
├── algos/          # RL algorithm implementations
└── tools/          # Training, analysis and visualization scripts

outputs/            # Training results and bias measurements
paper/              # LaTeX internship report
references/         # Academic papers
justfile           # Task automation commands
```

## Authors

- [Paul Chambaz](https://www.linkedin.com/in/paul-chambaz-17235a158/)
- Supervised by [Olivier Sigaud](https://www.isir.upmc.fr/personnel/sigaud/) (ISIR)
- Academic Referee: [Thibaut Lust](https://www.lip6.fr/actualite/personnes-fiche.php?ident=P1372) (LIP6)

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
