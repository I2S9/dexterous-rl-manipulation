# Component Ablation Study

## Overview

This ablation study systematically isolates the impact of key training components:
- **Curriculum Learning**: Progressive difficulty increase
- **Dense Reward Shaping**: Dense reward signals vs sparse rewards

## Configurations Tested

1. **Baseline**: Curriculum + Dense Reward (full system)
2. **No Curriculum**: Fixed difficulty + Dense Reward
3. **No Dense Reward**: Curriculum + Sparse Reward
4. **Minimal**: Fixed difficulty + Sparse Reward (baseline minimal)

## Methodology

- Multiple random seeds (5 seeds) for robust statistics
- 200 episodes per configuration per seed
- Metrics tracked:
  - Final success rate
  - Mean episode length
  - Convergence step (first episode with success rate >= 0.5 over window)
  - Episode-by-episode success rates

## Results Structure

The ablation study produces:
- Statistical comparison across configurations
- Individual component impact quantification
- Combined impact analysis
- Visualizations of performance differences

## Key Metrics

### Final Success Rate
Comparison of final success rates across all configurations, showing the contribution of each component.

### Convergence Speed
Number of episodes required to reach a stable success rate, indicating learning efficiency.

### Component Impact
Individual contribution of:
- Curriculum learning to overall performance
- Dense reward shaping to overall performance
- Combined effect of both components

## Usage

Run the complete ablation study:
```bash
python evaluation/run_component_ablation.py
```

This will:
1. Train all configurations with multiple seeds
2. Compute statistics across seeds
3. Generate comparison report
4. Create visualizations

## Output Files

- `logs/component_ablation_results.json`: Complete results data
- `logs/component_ablation.png`: Visualization plots

## Interpretation

The ablation study provides quantitative justification for design choices:
- If curriculum learning shows significant improvement, it validates the curriculum approach
- If dense reward shows significant improvement, it validates the reward shaping approach
- Combined impact shows the total benefit of the full system over minimal baseline

## Validation

The system validates that:
- All configurations are tested
- Statistics are computed correctly across seeds
- Differences between configurations are measurable
- Results are reproducible with fixed seeds
