# 23-Feature Architecture Implementation

## Overview

The 23-feature architecture represents a significant improvement over the previous 51/57-feature approach. Instead of trying to generate all features within the GAN, we focus on generating only the 23 core base features and calculate technical indicators and datetime features as post-processing.

## Key Benefits

1. **Authenticity**: The GAN focuses on learning the core relationships between the 23 base features
2. **Computational Efficiency**: Smaller networks train faster and require less memory  
3. **Better Learning**: The discriminator can focus on distinguishing realistic vs fake patterns in core features
4. **Deterministic Post-Processing**: Technical indicators calculated from the 23 features will be mathematically correct
5. **No Expansion Complexity**: Eliminates the problematic feature expansion methods

## Architecture Changes

### Generator (23 features output)
- **Input**: Noise (100-dim), Context (64-dim), Conditions (10-dim)
- **Output**: (batch_size, 144, 23) - sequences of 23 base features
- **Process**: BiLSTM Z-generator → VAE Decoder → 23 base features directly

### Discriminator (23 features input)  
- **Input**: (batch_size, 144, 23)
- **Architecture**: Conv1D (23→32→16→8) → BiLSTM (8→64) → Dense (64→16→8→1)
- **Output**: (batch_size, 1) - real/fake probability

## 23 Base Features

Based on the VAE decoder output feature names from config.py:

1. **OHLC Core (4 features)**:
   - OPEN, HIGH, LOW, CLOSE

2. **Market Data (3 features)**:
   - vix_close, S&P500_Close  
   - BC-BO, BH-BL (bid/ask spreads)

3. **Sub-periodicity Ticks (16 features)**:
   - CLOSE_15m_tick_1 through CLOSE_15m_tick_8 (8 features)
   - CLOSE_30m_tick_1 through CLOSE_30m_tick_8 (8 features)

## Post-Processing Pipeline

After the GAN generates the 23 base features, we apply post-processing to calculate additional features:

### 1. Technical Indicators (calculated from OHLC)
```python
def calculate_technical_indicators(ohlc_data):
    """Calculate technical indicators from OHLC data."""
    # RSI (14-period)
    rsi = calculate_rsi(ohlc_data['CLOSE'], period=14)
    
    # MACD (12, 26, 9)
    macd_line, macd_signal, macd_histogram = calculate_macd(ohlc_data['CLOSE'])
    
    # EMA (various periods)
    ema_20 = calculate_ema(ohlc_data['CLOSE'], period=20)
    ema_50 = calculate_ema(ohlc_data['CLOSE'], period=50)
    
    # Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic(ohlc_data, k_period=14, d_period=3)
    
    # ATR (Average True Range)
    atr = calculate_atr(ohlc_data, period=14)
    
    # Additional indicators as needed...
    
    return {
        'RSI': rsi,
        'MACD': macd_line,
        'MACD_Signal': macd_signal,
        'MACD_Histogram': macd_histogram,
        'EMA_20': ema_20,
        'EMA_50': ema_50,
        'Stochastic_K': stoch_k,
        'Stochastic_D': stoch_d,
        'ATR': atr,
        # ... more indicators
    }
```

### 2. Datetime Features (cyclical encoding)
```python
def calculate_datetime_features(timestamps):
    """Calculate cyclical datetime features."""
    # Extract datetime components
    hours = timestamps.hour
    days_of_week = timestamps.dayofweek
    days_of_month = timestamps.day
    months = timestamps.month
    days_of_year = timestamps.dayofyear
    
    # Convert to cyclical features using sin/cos
    datetime_features = {
        'hour_of_day_sin': np.sin(2 * np.pi * hours / 24),
        'hour_of_day_cos': np.cos(2 * np.pi * hours / 24),
        'day_of_week_sin': np.sin(2 * np.pi * days_of_week / 7),
        'day_of_week_cos': np.cos(2 * np.pi * days_of_week / 7),
        'day_of_month_sin': np.sin(2 * np.pi * days_of_month / 31),
        'day_of_month_cos': np.cos(2 * np.pi * days_of_month / 31),
        'month_sin': np.sin(2 * np.pi * months / 12),
        'month_cos': np.cos(2 * np.pi * months / 12),
        'day_of_year_sin': np.sin(2 * np.pi * days_of_year / 366),
        'day_of_year_cos': np.cos(2 * np.pi * days_of_year / 366),
    }
    
    return datetime_features
```

### 3. Complete Post-Processing Function
```python
def post_process_generated_data(generated_23_features, start_timestamp):
    """
    Complete post-processing pipeline for generated 23-feature data.
    
    Args:
        generated_23_features: (n_samples, sequence_length, 23) array
        start_timestamp: Starting timestamp for the sequences
        
    Returns:
        Dictionary with all features including technical indicators and datetime features
    """
    n_samples, seq_len, _ = generated_23_features.shape
    
    all_features = {}
    
    # 1. Extract base features
    all_features['OPEN'] = generated_23_features[:, :, 0]
    all_features['HIGH'] = generated_23_features[:, :, 1] 
    all_features['LOW'] = generated_23_features[:, :, 2]
    all_features['CLOSE'] = generated_23_features[:, :, 3]
    # ... extract other base features
    
    # 2. Calculate technical indicators for each sample
    for i in range(n_samples):
        ohlc_data = {
            'OPEN': all_features['OPEN'][i],
            'HIGH': all_features['HIGH'][i],
            'LOW': all_features['LOW'][i], 
            'CLOSE': all_features['CLOSE'][i]
        }
        
        # Calculate technical indicators
        tech_indicators = calculate_technical_indicators(ohlc_data)
        
        # Store in all_features (initialize arrays if first iteration)
        for indicator_name, values in tech_indicators.items():
            if indicator_name not in all_features:
                all_features[indicator_name] = np.zeros((n_samples, seq_len))
            all_features[indicator_name][i] = values
    
    # 3. Calculate datetime features
    timestamps = pd.date_range(start=start_timestamp, periods=seq_len, freq='1H')
    datetime_features = calculate_datetime_features(timestamps)
    
    # Broadcast datetime features to all samples
    for feature_name, values in datetime_features.items():
        all_features[feature_name] = np.tile(values, (n_samples, 1))
    
    return all_features
```

## Configuration Changes Made

### app/config.py
```python
# Updated discriminator configuration for 23 features
"num_features": 23,  # Updated: Use 23 base features instead of 51
```

### tsg_plugins/discriminator_plugin.py  
```python
# Updated input configuration
"num_features": 23,  # Updated: Use 23 base features instead of 51
```

### tsg_plugins/generator_plugin/generator_plugin.py
```python
# Updated target output features
"num_features": 23,  # Updated: Target output features (23 base features instead of 51)
```

### tsg_plugins/generator_plugin/_build_composite_generator.py
```python
# Removed feature expansion logic - output 23 features directly
# === CREATE COMPOSITE MODEL ===
composite_generator = Model(
    inputs=[noise_input, context_input, conditions_input],
    outputs=base_sequence,  # Direct output of 23 features
    name="composite_gan_generator"
)
```

## Training Process

1. **Generator Training**: Learns to generate realistic 23-feature sequences
2. **Discriminator Training**: Learns to distinguish real vs fake 23-feature sequences  
3. **Post-Processing**: Applied after generation to create full feature set as needed

## Benefits Over Previous Architecture

1. **No Feature Expansion Errors**: Eliminates TensorFlow compatibility issues
2. **Faster Training**: Smaller networks with fewer parameters
3. **Better Quality**: GAN focuses on core feature relationships
4. **Flexibility**: Post-processing can be customized based on requirements
5. **Maintainability**: Cleaner architecture without complex expansion logic

## Migration Path

For existing workflows that expect 51+ features:

1. Generate 23-feature data using the new architecture
2. Apply post-processing to calculate additional features as needed
3. Use the expanded feature set for downstream tasks

This approach maintains compatibility while providing the benefits of the simpler 23-feature architecture.
