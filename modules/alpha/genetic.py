"""
Genetic Algorithm Optimizer - Strategy Evolution Engine
Author: Erdinc Erdogan
Purpose: Implements genetic algorithm optimization for evolving trading strategies through
natural selection, crossover, and mutation of strategy parameters.
References:
- Genetic Algorithms: Holland (1975) "Adaptation in Natural and Artificial Systems"
- Evolutionary Computation in Finance
- Multi-Objective Optimization (NSGA-II)
Usage:
    optimizer = GeneticOptimizer(population_size=100, mutation_rate=0.1)
    best_strategy = optimizer.evolve(fitness_function, generations=50)
"""
import os
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from colorama import Fore, Style


@dataclass
class Gene:
    """Strategy gene."""
    name: str
    value: float
    min_val: float
    max_val: float
    mutation_rate: float = 0.1


@dataclass
class Chromosome:
    """Strategy chromosome (set of genes)."""
    genes: List[Gene]
    fitness: float = 0.0
    generation: int = 0
    
    def to_strategy(self) -> Dict:
        """Convert chromosome to strategy parameters."""
        return {gene.name: gene.value for gene in self.genes}


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm Optimizer.
    
    Optimize strategy parameters by evolving them:
    - RSI period, Stop-Loss ratio, Position Size, etc.
    - Select the best "offspring" each generation
    - Generate new strategies through crossover and mutation
    """
    
    # Default gene pool
    DEFAULT_GENES = [
        Gene("rsi_period", 14, 5, 50),
        Gene("rsi_overbought", 70, 60, 90),
        Gene("rsi_oversold", 30, 10, 40),
        Gene("sma_fast", 20, 5, 50),
        Gene("sma_slow", 50, 20, 200),
        Gene("stop_loss_pct", 0.02, 0.005, 0.10),
        Gene("take_profit_pct", 0.05, 0.01, 0.20),
        Gene("position_size_pct", 0.10, 0.01, 0.30),
        Gene("max_drawdown_pct", 0.15, 0.05, 0.30),
        Gene("volatility_filter", 0.02, 0.005, 0.05),
    ]
    
    def __init__(self,
                 population_size: int = 50,
                 elite_ratio: float = 0.1,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        """
        Args:
            population_size: Population size
            elite_ratio: Elite ratio (directly passed)
            mutation_rate: Mutation rate
            crossover_rate: Crossover rate
        """
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.best_chromosome = None
    
    def initialize_population(self, genes: List[Gene] = None) -> List[Chromosome]:
        """
        Initialize the starting population.
        """
        genes = genes or self.DEFAULT_GENES
        
        self.population = []
        
        for _ in range(self.population_size):
            # Random gene values for each individual
            chromosome_genes = []
            for gene in genes:
                new_gene = Gene(
                    name=gene.name,
                    value=random.uniform(gene.min_val, gene.max_val),
                    min_val=gene.min_val,
                    max_val=gene.max_val,
                    mutation_rate=gene.mutation_rate
                )
                chromosome_genes.append(new_gene)
            
            chromosome = Chromosome(genes=chromosome_genes, generation=0)
            self.population.append(chromosome)
        
        print(f"{Fore.CYAN}🧬 Population initialized: {self.population_size} individuals{Style.RESET_ALL}", flush=True)
        
        return self.population
    
    def evaluate_fitness(self, 
                        chromosome: Chromosome,
                        backtest_func: Callable,
                        price_data: List[float]) -> float:
        """
        Fitness evaluation (backtest).
        
        Args:
            chromosome: Chromosome to evaluate
            backtest_func: Backtest function
            price_data: Price data
        """
        strategy = chromosome.to_strategy()
        
        try:
            # Run backtest
            result = backtest_func(strategy, price_data)
            
            # Calculate fitness (Sharpe Ratio based)
            sharpe = result.get("sharpe_ratio", 0)
            total_return = result.get("return_pct", 0)
            max_drawdown = result.get("max_drawdown", 1)
            
            # Multi-objective fitness
            fitness = (
                sharpe * 0.4 +
                total_return * 0.3 +
                (1 - max_drawdown) * 0.3
            )
            
            # Penalize negative returns
            if total_return < 0:
                fitness *= 0.5
            
        except Exception as e:
            fitness = -1.0
        
        chromosome.fitness = fitness
        return fitness
    
    def evaluate_population(self, 
                           backtest_func: Callable,
                           price_data: List[float]) -> None:
        """Evaluate the entire population."""
        print(f"{Fore.CYAN}📊 Evaluating generation {self.generation}...{Style.RESET_ALL}", flush=True)
        
        for chromosome in self.population:
            self.evaluate_fitness(chromosome, backtest_func, price_data)
        
        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)
        
        # Save the best
        if self.best_chromosome is None or self.population[0].fitness > self.best_chromosome.fitness:
            self.best_chromosome = self.population[0]
        
        self.best_fitness_history.append(self.population[0].fitness)
        
        print(f"{Fore.GREEN}  → Best fitness: {self.population[0].fitness:.4f}{Style.RESET_ALL}", flush=True)
    
    def select_parents(self) -> Tuple[Chromosome, Chromosome]:
        """
        Parent selection (Tournament Selection).
        """
        tournament_size = 5
        
        # Tournament 1
        candidates1 = random.sample(self.population, tournament_size)
        parent1 = max(candidates1, key=lambda c: c.fitness)
        
        # Tournament 2
        candidates2 = random.sample(self.population, tournament_size)
        parent2 = max(candidates2, key=lambda c: c.fitness)
        
        return parent1, parent2
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Crossover.
        
        Mixes genes of two parents to produce children.
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Uniform crossover
        child1_genes = []
        child2_genes = []
        
        for g1, g2 in zip(parent1.genes, parent2.genes):
            if random.random() < 0.5:
                child1_genes.append(Gene(g1.name, g1.value, g1.min_val, g1.max_val))
                child2_genes.append(Gene(g2.name, g2.value, g2.min_val, g2.max_val))
            else:
                child1_genes.append(Gene(g1.name, g2.value, g1.min_val, g1.max_val))
                child2_genes.append(Gene(g2.name, g1.value, g2.min_val, g2.max_val))
        
        child1 = Chromosome(genes=child1_genes, generation=self.generation + 1)
        child2 = Chromosome(genes=child2_genes, generation=self.generation + 1)
        
        return child1, child2
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutation.
        
        Randomly alters genes.
        """
        for gene in chromosome.genes:
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation_strength = (gene.max_val - gene.min_val) * 0.1
                gene.value += random.gauss(0, mutation_strength)
                
                # Keep within bounds
                gene.value = max(gene.min_val, min(gene.max_val, gene.value))
        
        return chromosome
    
    def evolve(self, generations: int = 50,
              backtest_func: Callable = None,
              price_data: List[float] = None) -> Dict:
        """
        Evolution loop.
        
        Args:
            generations: Number of generations
            backtest_func: Backtest function (simulated if None)
            price_data: Price data
        """
        print(f"{Fore.CYAN}🧬 GENETIC ALGORITHM EVOLUTION STARTING{Style.RESET_ALL}", flush=True)
        print(f"   Population: {self.population_size}, Generations: {generations}", flush=True)
        
        # Create population
        if not self.population:
            self.initialize_population()
        
        # Simulate if no backtest function
        if backtest_func is None:
            backtest_func = self._simulated_backtest
        
        if price_data is None:
            price_data = self._generate_price_data()
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate
            self.evaluate_population(backtest_func, price_data)
            
            # Create new generation
            new_population = []
            
            # Elitism: Directly transfer the best individuals
            elite_count = int(self.population_size * self.elite_ratio)
            new_population.extend(self.population[:elite_count])
            
            # Fill the rest with crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population
            
            # Progress
            if (gen + 1) % 10 == 0 or gen == generations - 1:
                print(f"{Fore.GREEN}   Generation {gen + 1}/{generations}: Fitness = {self.best_chromosome.fitness:.4f}{Style.RESET_ALL}", flush=True)
        
        print(f"{Fore.GREEN}🏆 EVOLUTION COMPLETED!{Style.RESET_ALL}", flush=True)
        
        return self.get_best_strategy()
    
    def _simulated_backtest(self, strategy: Dict, price_data: List[float]) -> Dict:
        """Simulated backtest."""
        # Simple simulation instead of real backtest
        
        # Strategy parameters
        rsi_period = int(strategy.get("rsi_period", 14))
        stop_loss = strategy.get("stop_loss_pct", 0.02)
        take_profit = strategy.get("take_profit_pct", 0.05)
        
        # Simple performance simulation
        # Better parameters yield better results
        
        base_return = 0.1
        
        # Proximity to optimal RSI values
        rsi_optimal = 14
        rsi_penalty = abs(rsi_period - rsi_optimal) / 50
        
        # Stop-loss / Take-profit ratio
        risk_reward = take_profit / stop_loss if stop_loss > 0 else 1
        rr_bonus = min(risk_reward / 3, 0.1)
        
        # Add randomness
        noise = random.gauss(0, 0.05)
        
        total_return = base_return - rsi_penalty + rr_bonus + noise
        sharpe = total_return * 5 + random.gauss(0, 0.3)
        max_dd = max(0.05, stop_loss * 3 + random.uniform(0, 0.1))
        
        return {
            "return_pct": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd
        }
    
    def _generate_price_data(self, length: int = 252) -> List[float]:
        """Generate synthetic price data."""
        prices = [100]
        for _ in range(length - 1):
            change = random.gauss(0.0002, 0.02)  # Daily ~0.02% mean, 2% std
            prices.append(prices[-1] * (1 + change))
        return prices
    
    def get_best_strategy(self) -> Dict:
        """Return the best strategy."""
        if not self.best_chromosome:
            return {"error": "Evolution not performed"}
        
        strategy = self.best_chromosome.to_strategy()
        
        return {
            "strategy": strategy,
            "fitness": self.best_chromosome.fitness,
            "generation": self.best_chromosome.generation,
            "improvement": (
                (self.best_fitness_history[-1] - self.best_fitness_history[0]) / abs(self.best_fitness_history[0]) * 100
                if self.best_fitness_history and self.best_fitness_history[0] != 0 else 0
            )
        }
    
    def generate_evolution_report(self) -> str:
        """Evolution report."""
        best = self.get_best_strategy()
        
        report = f"""
<genetic_algorithm>
🧬 GENETIC ALGORITHM OPTIMIZATION REPORT
════════════════════════════════════════

📊 EVOLUTION RESULT:
  • Generation: {self.generation}
  • Best Fitness: {best.get('fitness', 0):.4f}
  • Improvement: %{best.get('improvement', 0):.1f}

🏆 OPTIMUM STRATEGY PARAMETERS:
"""
        if "strategy" in best:
            for name, value in best["strategy"].items():
                report += f"  • {name}: {value:.4f}\n"
        
        report += f"""
📈 FITNESS HISTORY:
  • Start: {self.best_fitness_history[0]:.4f if self.best_fitness_history else 'N/A'}
  • End: {self.best_fitness_history[-1]:.4f if self.best_fitness_history else 'N/A'}

💡 NOTE: The bot continuously evolves to optimize itself.
</genetic_algorithm>
"""
        return report


# Macro module init
__all__ = ['GeneticAlgorithmOptimizer', 'Gene', 'Chromosome']
