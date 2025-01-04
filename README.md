# Cerebrix - BCI AI Controller API

Cerebrix is a Brain-Computer Interface (BCI) AI Controller API that enables developers to create EEG-based actions through recording, preprocessing, and training on EEG signals to identify action intents. This framework makes it easier to integrate brainwave-driven functions into applications.

## Core Features

### Dynamic Window Processing
- Creates 200ms windows (5 frames at 25Hz) with temporal context
- Each window includes:
  - Up to 2 past frames (zero-padded if unavailable)
  - Current frame
  - Up to 2 future frames (zero-padded if unavailable)
- Maintains temporal relationships and proper padding
- Example progression for 10-frame sample:
  - Frame 0: 2 zero-padded past + current + 2 actual future
  - Frame 1: 1 zero-padded past + 1 actual past + current + 2 actual future
  - Frame 5: 2 actual past + current + 2 actual future
  - Frame 8: 2 actual past + current + 1 actual future + 1 zero-padded future
  - Frame 9: 2 actual past + current + 2 zero-padded future

### Feature Extraction Pipeline

#### Handcrafted Features
Domain-specific EEG feature extraction:
- **Frequency Features**
  - Band powers (delta, theta, alpha, beta, gamma)
  - Spectral ratios and relationships
  - Spectral entropy for complexity analysis
- **Temporal Features**
  - Hjorth parameters (activity, mobility, complexity)
  - Statistical measures (mean, std, peak-to-peak)
  - Stability metrics across time
- **Spatial Features**
  - Channel connectivity patterns
  - Cross-correlations between channels
  - Spatial distribution analysis

#### Unsupervised Feature Learning
Multiple approaches for automatic pattern discovery:
- **Autoencoder**
  - Compresses EEG data into dense representations
  - Captures global signal characteristics
  - Learns temporal-spatial structure patterns
  
- **Deep Embedded Clustering (DEC)**
  - Identifies distinct brain state clusters
  - Creates cluster membership features
  - Maps signal patterns to brain states
  
- **Recurrent Autoencoder**
  - Specialized for temporal dependencies
  - LSTM-based sequence understanding
  - Enhanced temporal dynamics capture
  
- **Self-Organizing Map (SOM)**
  - Topological mapping of brain states
  - Preserves signal neighborhood relationships
  - Position-based feature generation

### Model Architectures

#### Raw Features Transformer
- Parallel processing of raw signals and extracted features
- Cross-attention mechanism for modality fusion
- Transformer decoder for sequence understanding
- Benefits:
  - Long-range dependency capture
  - Variable sequence length handling
  - Built-in attention mechanisms

#### Hierarchical Attention Network (HAN)
- Multi-level attention processing
- Separate pathways for raw and feature data
- Hierarchical fusion of modalities
- Benefits:
  - Structured feature hierarchy handling
  - Interpretable attention weights
  - Feature importance learning

#### Graph Neural Network (GNN)
- Represents EEG as connected graph
- Learns channel relationships
- Spatial-temporal pattern processing
- Benefits:
  - Complex relationship modeling
  - Non-linear dependency capture
  - Natural spatial pattern learning

#### CNN-LSTM Baseline
- Direct raw signal processing
- Spatial-temporal feature learning
- Sequential pattern recognition
- Benefits:
  - Simple architecture
  - End-to-end learning
  - Basic feature discovery

### Architecture-Feature Combinations

#### 1. Raw Features Transformer Combinations

A. With Full Feature Set
- **Implementation**: Combines raw EEG with all feature types (Handcrafted + Unsupervised)
- **Processing Flow**:
  - Raw data → CNN processing
  - Features → Dense embedding
  - Cross-attention fusion
  - Transformer decoder for final decision
- **Strengths**:
  - Comprehensive pattern capture
  - Strong temporal relationship modeling
  - Effective feature-signal correlation
- **Best For**: Complex action recognition requiring both signal and feature patterns

B. With Handcrafted Features Only
- **Implementation**: Raw EEG + domain-specific features
- **Processing Flow**:
  - Raw signals → Spatial-temporal encoding
  - Handcrafted features → Feature embedding
  - Cross-modal attention fusion
- **Strengths**:
  - Domain knowledge integration
  - Interpretable feature relationships
  - Efficient processing
- **Best For**: Well-understood action patterns with known markers

C. With Unsupervised Features Only
- **Implementation**: Raw EEG + learned representations
- **Processing Flow**:
  - Raw data → Direct processing
  - Unsupervised features → Latent embedding
  - Attention-based combination
- **Strengths**:
  - Automatic pattern discovery
  - Novel relationship detection
  - Adaptive feature learning
- **Best For**: Complex patterns without clear domain markers

#### 2. Hierarchical Attention Network Combinations

A. Full Feature Integration
- **Implementation**: Multi-level attention across all data types
- **Processing Levels**:
  1. Channel-level attention
  2. Temporal-level attention
  3. Feature-level attention
  4. Cross-modal fusion attention
- **Strengths**:
  - Structured feature processing
  - Clear attention hierarchies
  - Interpretable importance weights
- **Best For**: Actions with multi-level patterns

B. Feature-Focused Processing
- **Implementation**: Hierarchical processing of feature sets
- **Attention Levels**:
  1. Individual feature attention
  2. Feature group attention
  3. Feature interaction attention
- **Strengths**:
  - Deep feature understanding
  - Clear feature importance
  - Efficient processing
- **Best For**: Feature-rich action recognition

#### 3. Graph Neural Network Combinations

A. Channel-Feature Graph
- **Implementation**: Graph with both channel and feature nodes
- **Graph Structure**:
  - Nodes: EEG channels + feature vectors
  - Edges: Signal + feature relationships
- **Strengths**:
  - Complex relationship modeling
  - Spatial-feature integration
  - Dynamic graph structure
- **Best For**: Actions with strong spatial patterns

B. Feature Relationship Graph
- **Implementation**: Graph based on feature relationships
- **Structure**:
  - Nodes: Feature vectors
  - Edges: Feature correlations
- **Strengths**:
  - Feature interaction modeling
  - Relationship discovery
  - Pattern visualization
- **Best For**: Complex feature interactions

#### 4. CNN-LSTM Raw Processing

A. Pure Signal Processing
- **Implementation**: Direct raw EEG processing
- **Architecture**:
  - CNN layers for spatial features
  - LSTM for temporal patterns
- **Strengths**:
  - End-to-end learning
  - No feature engineering needed
  - Natural sequence processing
- **Best For**: Simple action patterns

B. Signal + Feature Enhancement
- **Implementation**: Raw processing with feature injection
- **Processing Flow**:
  - CNN → Spatial features
  - Feature concatenation
  - LSTM temporal processing
- **Strengths**:
  - Enhanced pattern detection
  - Feature-guided learning
  - Balanced approach
- **Best For**: Mixed pattern recognition

#### 5. Feature-Only Architectures

A. Transformer with Combined Features
- **Implementation**: All features → Transformer
- **Strengths**:
  - Pure feature-based decisions
  - Strong feature relationships
  - Efficient processing
- **Best For**: Feature-rich, well-defined patterns

B. HAN with Feature Hierarchy
- **Implementation**: Hierarchical feature processing
- **Strengths**:
  - Structured feature analysis
  - Clear importance weights
  - Interpretable decisions
- **Best For**: Complex feature interactions

C. GNN with Feature Graphs
- **Implementation**: Graph-based feature processing
- **Strengths**:
  - Complex feature relationships
  - Non-linear interactions
  - Pattern discovery
- **Best For**: Feature relationship mapping

#### Performance Considerations

- **Computational Efficiency**:
  - Raw Features Transformer: High computation
  - HAN: Medium computation
  - GNN: Medium-High computation
  - CNN-LSTM: Medium computation
  - Feature-Only: Low computation

- **Memory Usage**:
  - Full Feature Integration: High
  - Raw Processing: Medium
  - Feature-Only: Low

- **Training Time**:
  - Complex architectures (Transformer, GNN): Longer
  - Simpler architectures (CNN-LSTM): Shorter
  - Feature-only: Fastest

- **Inference Speed**:
  - Feature-only: Fastest
  - Raw processing: Medium
  - Combined approaches: Slower
