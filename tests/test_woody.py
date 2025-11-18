#%%
import numpy as np
from scipy.signal import butter, filtfilt, detrend, correlate
import matplotlib.pyplot as plt

def create_mock_erp_dataset(
    n_trials: int = 100,
    n_samples: int = 1000,
    sfreq: float = 500.0,
    epoch_twd_ms: list[float] = [-200, 800],
    peak_time_range_ms: list[float] = [280, 320],
    snr: float = 2.0,
    random_seed: int = 42
):
    """
    Create mock ERP dataset with varied peak timings for testing Woody's algorithm.

    Parameters:
    -----------
    n_trials : int
        Number of trials to generate
    n_samples : int
        Number of time samples per trial
    sfreq : float
        Sampling frequency in Hz
    epoch_twd_ms : list[float]
        Time window [start, end] in milliseconds
    peak_time_range_ms : list[float]
        Range [min, max] for peak timing in milliseconds
    snr : float
        Signal-to-noise ratio
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict with 'data', 'true_peak_times_ms', 'time_axis_ms'
    """
    np.random.seed(random_seed)

    # Create time axis
    time_axis_ms = np.linspace(epoch_twd_ms[0], epoch_twd_ms[1], n_samples)

    # Generate random peak times for each trial within the specified range
    true_peak_times_ms = np.random.uniform(peak_time_range_ms[0], peak_time_range_ms[1], n_trials)

    # Calculate reference peak time (center of range) for latency calculation
    reference_peak_time_ms = np.mean(peak_time_range_ms)
    true_latencies_ms = true_peak_times_ms - reference_peak_time_ms

    # Generate trials with varying peak timings
    data = np.zeros((n_samples, n_trials))

    for trial_idx in range(n_trials):
        # Get the peak time for this trial
        peak_time_ms = true_peak_times_ms[trial_idx]
        peak_idx = np.argmin(np.abs(time_axis_ms - peak_time_ms))

        # Create ERP waveform for this trial with peak at the specified time
        trial_erp = np.zeros(n_samples)

        # P300 component (positive peak at the sampled time)
        p300_width = int(0.1 * sfreq)  # 100ms width
        p300_indices = np.arange(max(0, peak_idx - p300_width//2),
                                min(n_samples, peak_idx + p300_width//2))
        if len(p300_indices) > 0:
            trial_erp[p300_indices] = 5 * np.exp(-0.5 * ((p300_indices - peak_idx) / (p300_width/4))**2)

        # N200 component (negative peak 100ms before P300)
        n200_time_ms = peak_time_ms - 100
        n200_idx = np.argmin(np.abs(time_axis_ms - n200_time_ms))
        n200_width = int(0.08 * sfreq)  # 80ms width
        n200_indices = np.arange(max(0, n200_idx - n200_width//2),
                                min(n_samples, n200_idx + n200_width//2))
        if len(n200_indices) > 0:
            trial_erp[n200_indices] -= 3 * np.exp(-0.5 * ((n200_indices - n200_idx) / (n200_width/4))**2)

        # Add realistic noise
        noise = np.random.normal(0, np.max(np.abs(trial_erp)) / snr, n_samples)

        # Add some trial-to-trial amplitude variability
        amplitude_factor = np.random.uniform(0.7, 1.3)

        # Add baseline drift
        baseline_drift = np.random.uniform(-1, 1)

        # Combine components
        data[:, trial_idx] = amplitude_factor * trial_erp + noise + baseline_drift

    # Create a reference template at the center peak time for comparison
    reference_peak_idx = np.argmin(np.abs(time_axis_ms - reference_peak_time_ms))
    template_erp = np.zeros(n_samples)

    # P300 at reference time
    p300_width = int(0.1 * sfreq)
    p300_indices = np.arange(max(0, reference_peak_idx - p300_width//2),
                            min(n_samples, reference_peak_idx + p300_width//2))
    if len(p300_indices) > 0:
        template_erp[p300_indices] = 5 * np.exp(-0.5 * ((p300_indices - reference_peak_idx) / (p300_width/4))**2)

    # N200 at reference time
    ref_n200_time_ms = reference_peak_time_ms - 100
    ref_n200_idx = np.argmin(np.abs(time_axis_ms - ref_n200_time_ms))
    n200_width = int(0.08 * sfreq)
    n200_indices = np.arange(max(0, ref_n200_idx - n200_width//2),
                            min(n_samples, ref_n200_idx + n200_width//2))
    if len(n200_indices) > 0:
        template_erp[n200_indices] -= 3 * np.exp(-0.5 * ((n200_indices - ref_n200_idx) / (n200_width/4))**2)

    return {
        'data': data,
        'true_peak_times_ms': true_peak_times_ms,
        'true_latencies_ms': true_latencies_ms,
        'time_axis_ms': time_axis_ms,
        'template_erp': template_erp,
        'reference_peak_time_ms': reference_peak_time_ms
    }

def woodys_algorithm(
    X: np.ndarray,
    sfreq: float,
    time_window_ms: list[float],
    epoch_twd_ms: list[float] = [-200, 800],
    max_iter: int = 20,
    convergence_threshold_ms: float = 1.0
):
    """
    Implementation of Woody's adaptive filter algorithm for ERP analysis.
    See prompt for full docstring.
    """
    data = X.T
    n_samples, n_trials = data.shape
    samp_interval_ms = 1000.0 / sfreq
    def ms_to_samples(ms):
        return int((ms - epoch_twd_ms[0]) / samp_interval_ms)
    win_start, win_end = ms_to_samples(time_window_ms[0]), ms_to_samples(time_window_ms[1])
    time_window_slice = slice(win_start, win_end)
    original_erp = np.mean(data, axis=1)
    template = original_erp.copy()
    latencies_samples = np.zeros(n_trials, dtype=int)
    for i in range(max_iter):
        prev_latencies_samples = latencies_samples.copy()
        template_segment = template[time_window_slice]
        for trial_idx in range(n_trials):
            trial_segment = data[time_window_slice, trial_idx]
            xcorr = correlate(
                trial_segment - trial_segment.mean(),
                template_segment - template_segment.mean(),
                mode='same'
            )
            center_idx = len(xcorr) // 2
            peak_idx = np.argmax(xcorr)
            latencies_samples[trial_idx] = peak_idx - center_idx
        latencies_samples -= np.round(np.median(latencies_samples)).astype(int)
        aligned_trials = np.zeros_like(data)
        for trial_idx in range(n_trials):
            aligned_trials[:, trial_idx] = np.roll(data[:, trial_idx], -latencies_samples[trial_idx])
        template = np.mean(aligned_trials, axis=1)
        latency_change_samples = np.sum(np.abs(latencies_samples - prev_latencies_samples))
        latency_change_ms = latency_change_samples * samp_interval_ms
        if latency_change_ms < convergence_threshold_ms:
            break
    aligned_erp = template
    latencies_ms = latencies_samples * samp_interval_ms
    return {
        'aligned_erp': aligned_erp,
        'original_erp': original_erp,
        'latencies_ms': latencies_ms
    }

def ride_decomposition(
    X: np.ndarray,
    sfreq: float,
    epoch_twd_ms: list[float] = [-200, 800],
    component_twd_ms: dict = None,
    high_cutoff_hz: float = 30.0
):
    """
    Barebones Python implementation of the Residue Iteration Decomposition (RIDE) algorithm.
    Now accepts epoch and component windows as arguments.

    This function decomposes single-channel EEG data into three components with
    variable single-trial latencies. It is designed to separate overlapping
    neural components in Event-Related Potential (ERP) analysis.

    Args:
        X (np.ndarray): The input EEG data as a 2D NumPy array with
                        shape (K_trials, T_samples) - single channel already selected.
        sfreq (float): The sampling frequency of the data in Hz.
        epoch_twd_ms (list[float], optional): The time window for the epoch in
                                          milliseconds. Defaults to [-200, 800].
        component_twd_ms (dict, optional): The time windows for the components in
                                          milliseconds. If None, defaults to
                                          reasonable values for an S-C-R paradigm.
                                          Example format for S-C-R paradigm:
                                          {'S': [50, 200], 'C': [250, 500], 'R': [500, 750]}
        high_cutoff_hz (float, optional): The high cutoff frequency for the
                                          low-pass filter used during latency
                                          estimation. Defaults to 30.0 Hz.

    Returns:
        dict: A dictionary containing the results:
              - 'reconstructed_erp': The new ERP waveform, reconstructed by
                summing the stimulus-locked versions of the decomposed components.
              - 'original_erp': The classic ERP obtained by averaging all trials.
              - 'components': A (T, 3) array with the latency-aligned component waveforms.
              - 'latencies_ms': A (K, 3) array with the estimated single-trial
                latencies for each component in milliseconds.
    """
    # ----------------------------------------------------------------------
    # 1. Helper Functions
    # ----------------------------------------------------------------------

    def _shift_waveform(wave, shift_samples):
        """Shifts a 1D waveform by an integer number of samples."""
        return np.roll(wave, int(round(shift_samples)))

    def _design_filter(sfreq, high_cutoff):
        """Designs a low-pass Butterworth filter."""
        nyquist = 0.5 * sfreq
        normal_cutoff = high_cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        return b, a

    # ----------------------------------------------------------------------
    # 2. Initialization and Configuration
    # ----------------------------------------------------------------------
    if X.ndim != 2:
        raise ValueError("Input data `X` must be a 2D array of shape (trials, samples) for single channel.")

    # Transpose to (samples, trials) for easier processing
    data = X.T
    n_samples, n_trials = data.shape
    original_erp = np.mean(data, axis=1)

    # --- Configuration (derived from inputs and assumptions) ---
    epoch_start_ms, epoch_end_ms = epoch_twd_ms[0], epoch_twd_ms[1]

    # If no component_twd_ms provided, use defaults for S-C-R paradigm
    if component_twd_ms is None:
        component_twd_ms = {
            'S': [50, 200],
            'C': [250, 500],
            'R': [500, 750]
        }

    # Use only the components provided in component_twd_ms
    comp_names = list(component_twd_ms.keys())
    n_components = len(comp_names)

    # Convert ms to sample indices
    def ms_to_samples(ms):
        return int((ms - epoch_start_ms) * sfreq / 1000)

    twd_samp = {name: [ms_to_samples(t[0]), ms_to_samples(t[1])] for name, t in component_twd_ms.items()}
    dur_samp = {name: (t[1] - t[0]) // 2 for name, t in twd_samp.items()}

    # Initialize filter coefficients
    filter_b, filter_a = _design_filter(sfreq, high_cutoff_hz)

    # ----------------------------------------------------------------------
    # 3. Initial Latency Estimation (Woody's Method)
    # ----------------------------------------------------------------------
    latencies = np.zeros((n_trials, n_components), dtype=int)
    initial_template = original_erp.copy()

    for i, name in enumerate(comp_names):
        win_start, win_end = twd_samp[name][0], twd_samp[name][1]
        # Ensure window is within bounds
        win_start = max(0, min(win_start, n_samples-1))
        win_end = max(win_start+1, min(win_end, n_samples))
        win = slice(win_start, win_end)
        search_dur = dur_samp[name]
        template_segment = initial_template[win]

        for trial_idx in range(n_trials):
            trial_segment = data[win, trial_idx]
            # Skip if segments are too small for correlation
            if len(trial_segment) < 3 or len(template_segment) < 3:
                latencies[trial_idx, i] = 0
                continue
            # Cross-correlate trial with the template
            xcorr = correlate(trial_segment - trial_segment.mean(), template_segment - template_segment.mean(), mode='same')
            # Find the peak of the correlation
            search_start = max(0, len(xcorr)//2 - search_dur)
            search_end = min(len(xcorr), len(xcorr)//2 + search_dur)
            if search_end <= search_start:
                latencies[trial_idx, i] = 0
                continue
            peak_idx = np.argmax(xcorr[search_start:search_end])
            latencies[trial_idx, i] = peak_idx - search_dur

    # Center latencies around zero by subtracting the median
    latencies -= np.median(latencies, axis=0).astype(int)

    # ----------------------------------------------------------------------
    # 4. Main RIDE Iteration Loop
    # ----------------------------------------------------------------------
    outer_iterations = 4
    inner_iterations = 50  # Safeguard for the inner loop

    # Initialize component waveforms
    # comp: latency-aligned components
    # comp_sl: stimulus-locked components
    comp = np.zeros((n_samples, n_components))
    comp_sl = np.zeros((n_samples, n_components))

    for iter_num in range(outer_iterations):

        # --- a) RIDE Inner Loop: Decompose components based on current latencies ---
        for _ in range(inner_iterations):
            comp_old = comp.copy()
            for i in range(n_components):
                # Calculate residue by subtracting all other stimulus-locked components
                other_comps = np.delete(comp_sl, i, axis=1)
                residue = data - np.sum(other_comps, axis=1, keepdims=True)

                # Align residue trials based on component i's latency
                aligned_residue = np.zeros_like(residue)
                for trial_idx in range(n_trials):
                    aligned_residue[:, trial_idx] = _shift_waveform(residue[:, trial_idx], -latencies[trial_idx, i])

                # The new component is the average of the aligned residues
                comp[:, i] = np.mean(aligned_residue, axis=1)

                # The stimulus-locked version: shift each trial's component back by its latency
                unaligned_comp = np.zeros_like(residue)
                for trial_idx in range(n_trials):
                    unaligned_comp[:, trial_idx] = _shift_waveform(comp[:, i], latencies[trial_idx, i])
                comp_sl[:, i] = np.mean(unaligned_comp, axis=1)

            # Check for convergence of the inner loop
            if np.linalg.norm(comp - comp_old) < 1e-6:
                break

        # --- b) Re-estimate Latencies based on new component waveforms ---
        for i, name in enumerate(comp_names):
            template = comp[:, i]
            win_start, win_end = twd_samp[name][0], twd_samp[name][1]
            # Ensure window is within bounds
            win_start = max(0, min(win_start, n_samples-1))
            win_end = max(win_start+1, min(win_end, n_samples))
            win = slice(win_start, win_end)
            search_dur = dur_samp[name]

            # Low-pass filter the template for smoother cross-correlation
            template_filt = filtfilt(filter_b, filter_a, template)

            other_comps_indices = [idx for idx in range(n_components) if idx != i]

            for trial_idx in range(n_trials):
                # Calculate residual by subtracting other components
                trial_data = data[:, trial_idx]
                residual = trial_data.copy()
                for j in other_comps_indices:
                    comp_to_subtract = _shift_waveform(comp[:, j], latencies[trial_idx, j])
                    residual -= comp_to_subtract

                # Filter and detrend the residual within the time window
                residual_filt = filtfilt(filter_b, filter_a, residual)
                if win_end - win_start > 0:
                    residual_filt[win] = detrend(residual_filt[win])

                # Skip if segments are too small for correlation
                residual_segment = residual_filt[win] - residual_filt[win].mean()
                template_segment = template_filt[win] - template_filt[win].mean()

                if len(residual_segment) < 3 or len(template_segment) < 3:
                    latencies[trial_idx, i] = 0
                    continue

                # Cross-correlate residual with template
                xcorr = correlate(residual_segment, template_segment, mode='same')

                # Find new peak
                center = len(xcorr) // 2
                search_start = max(0, center - search_dur)
                search_end = min(len(xcorr), center + search_dur)
                if search_end <= search_start:
                    latencies[trial_idx, i] = 0
                    continue
                peak_idx = np.argmax(xcorr[search_start:search_end])
                latencies[trial_idx, i] = (search_start + peak_idx) - center

        # Re-center latencies after each outer iteration
        latencies -= np.median(latencies, axis=0).astype(int)

    # ----------------------------------------------------------------------
    # 5. Finalization and Output (Following MATLAB logic more closely)
    # ----------------------------------------------------------------------

    # Calculate baseline window (typically -200 to 0 ms pre-stimulus)
    baseline_start_ms = epoch_start_ms
    baseline_end_ms = min(0, epoch_end_ms)  # Up to stimulus onset
    baseline_start_idx = max(0, int((baseline_start_ms - epoch_start_ms) * sfreq / 1000))
    baseline_end_idx = min(n_samples, int((baseline_end_ms - epoch_start_ms) * sfreq / 1000))
    baseline_slice = slice(baseline_start_idx, baseline_end_idx)

    # Apply baseline correction to components
    for i in range(n_components):
        if baseline_end_idx > baseline_start_idx:
            baseline_mean = np.mean(comp_sl[:, i][baseline_slice])
            comp_sl[:, i] -= baseline_mean
            comp[:, i] -= baseline_mean

    # The reconstructed ERP is simply the average of the latency-aligned trials
    # This should give the same result as Woody's algorithm for single component
    aligned_trials = np.zeros_like(data)
    for trial_idx in range(n_trials):
        aligned_trials[:, trial_idx] = _shift_waveform(data[:, trial_idx], -latencies[trial_idx, 0])

    reconstructed_erp = np.mean(aligned_trials, axis=1)

    # Apply baseline correction to reconstructed ERP
    if baseline_end_idx > baseline_start_idx:
        baseline_mean = np.mean(reconstructed_erp[baseline_slice])
        reconstructed_erp -= baseline_mean

    # Convert latencies from samples to milliseconds for the output
    latencies_ms = latencies * (1000.0 / sfreq)

    return {
        'reconstructed_erp': reconstructed_erp,
        'original_erp': original_erp,
        'components': comp,
        'components_sl': comp_sl,  # Also return stimulus-locked components
        'latencies_ms': latencies_ms
    }

#%%
# Run the test
print("Testing Woody's Algorithm with Mock Data")
print("=" * 50)

# Create mock dataset
mock_data = create_mock_erp_dataset(
    n_trials=50,
    n_samples=500,
    sfreq=500.0,
    peak_time_range_ms=[200, 500],
    snr=2.0
)

# Extract data for algorithm
X = mock_data['data'].T  # Woody's algorithm expects (n_trials, n_samples)
sfreq = 500.0
time_window_ms = [150, 450]  # Window around P300 component
epoch_twd_ms = [-200, 800]

print(f"Input data shape: {X.shape}")
print(f"Peak time range: {mock_data['true_peak_times_ms'].min():.1f} to {mock_data['true_peak_times_ms'].max():.1f} ms")
print(f"Reference peak time: {mock_data['reference_peak_time_ms']:.1f} ms")
print(f"True peak times (first 10): {mock_data['true_peak_times_ms'][:10]}")
print(f"True latencies relative to reference (first 10): {mock_data['true_latencies_ms'][:10]}")

# Run Woody's algorithm
results = woodys_algorithm(
    X=X,
    sfreq=sfreq,
    time_window_ms=time_window_ms,
    epoch_twd_ms=epoch_twd_ms,
    max_iter=20,
    convergence_threshold_ms=1.0
)

# Extract results
estimated_latencies_ms = results['latencies_ms']
aligned_erp = results['aligned_erp']
original_erp = results['original_erp']

# Calculate performance metrics
true_latencies = mock_data['true_latencies_ms']

# Account for the fact that Woody's algorithm centers latencies around median
true_latencies_centered = true_latencies - np.median(true_latencies)

# Calculate correlation between true and estimated latencies
correlation = np.corrcoef(true_latencies_centered, estimated_latencies_ms)[0, 1]

# Calculate RMSE
rmse = np.sqrt(np.mean((true_latencies_centered - estimated_latencies_ms)**2))

print(f"\nResults:")
print(f"Correlation between true and estimated latencies: {correlation:.3f}")
print(f"RMSE of latency estimates: {rmse:.2f} ms")
print(f"Estimated latencies (first 10): {estimated_latencies_ms[:10]}")

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# Plot 1: Individual trials before averaging
time_axis = mock_data['time_axis_ms']
n_trials_to_plot = min(10, X.shape[0])
for i in range(n_trials_to_plot):
    axes[0, 0].plot(time_axis, X[i, :], alpha=0.4, color='gray', linewidth=0.8)
axes[0, 0].plot(time_axis, mock_data['template_erp'], 'k--', label='True Template', linewidth=2)
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude (μV)')
axes[0, 0].set_title(f'Individual Trials (first {n_trials_to_plot})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Original vs Aligned ERP
axes[0, 1].plot(time_axis, original_erp, label='Original ERP', alpha=0.7, linewidth=2)
axes[0, 1].plot(time_axis, aligned_erp, label='Aligned ERP', linewidth=2)
axes[0, 1].plot(time_axis, mock_data['template_erp'], '--', label='True Template', alpha=0.8, linewidth=2)
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Amplitude (μV)')
axes[0, 1].set_title('ERP Waveforms')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: True vs Estimated Latencies
axes[0, 2].scatter(true_latencies_centered, estimated_latencies_ms, alpha=0.6)
axes[0, 2].plot([-20, 20], [-20, 20], 'r--', label='Perfect correlation')
axes[0, 2].set_xlabel('True Latencies (ms)')
axes[0, 2].set_ylabel('Estimated Latencies (ms)')
axes[0, 2].set_title(f'Latency Recovery (r={correlation:.3f})')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Latency distribution
axes[1, 0].hist(true_latencies, bins=15, alpha=0.5, label='True', density=True)
axes[1, 0].hist(estimated_latencies_ms, bins=15, alpha=0.5, label='Estimated', density=True)
axes[1, 0].set_xlabel('Latency (ms)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Latency Distributions')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Residuals
residuals = estimated_latencies_ms - true_latencies_centered
axes[1, 1].scatter(true_latencies_centered, residuals, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('True Latencies (ms)')
axes[1, 1].set_ylabel('Residuals (ms)')
axes[1, 1].set_title(f'Residuals (RMSE={rmse:.2f}ms)')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Aligned individual trials
aligned_data = np.zeros_like(X)
for trial_idx in range(X.shape[0]):
    aligned_data[trial_idx, :] = np.roll(X[trial_idx, :], -int(estimated_latencies_ms[trial_idx] * sfreq / 1000))

for i in range(n_trials_to_plot):
    axes[1, 2].plot(time_axis, aligned_data[i, :], alpha=0.4, color='blue', linewidth=0.8)
axes[1, 2].plot(time_axis, aligned_erp, 'r-', label='Aligned Average', linewidth=2)
axes[1, 2].plot(time_axis, mock_data['template_erp'], 'k--', label='True Template', linewidth=2)
axes[1, 2].set_xlabel('Time (ms)')
axes[1, 2].set_ylabel('Amplitude (μV)')
axes[1, 2].set_title(f'Aligned Trials (first {n_trials_to_plot})')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
print("\n" + "="*60)
print("COMPARING RIDE vs WOODY'S ALGORITHM")
print("="*60)

# Create mock dataset - same as before
mock_data = create_mock_erp_dataset(
    n_trials=50,
    n_samples=500,
    sfreq=500.0,
    peak_time_range_ms=[300, 500],  # Updated to 300-500ms range
    snr=2.0
)

# Extract data
X = mock_data['data'].T  # (n_trials, n_samples)
sfreq = 500.0
time_axis = mock_data['time_axis_ms']
epoch_twd_ms = [-200, 800]

print(f"Peak time range: {mock_data['true_peak_times_ms'].min():.1f} to {mock_data['true_peak_times_ms'].max():.1f} ms")

# ================================
# 1. RUN WOODY'S ALGORITHM
# ================================
print("\nRunning Woody's Algorithm...")
woody_results = woodys_algorithm(
    X=X,
    sfreq=sfreq,
    time_window_ms=[250, 550],  # Window around P300 component
    epoch_twd_ms=epoch_twd_ms,
    max_iter=20,
    convergence_threshold_ms=1.0
)

# ================================
# 2. RUN RIDE ALGORITHM (SINGLE COMPONENT)
# ================================
print("Running RIDE Algorithm (single component)...")
# Define single component window matching the peak range
ride_results = ride_decomposition(
    X=X,
    sfreq=sfreq,
    epoch_twd_ms=epoch_twd_ms,
    component_twd_ms={'S': [-200, 800]},
    high_cutoff_hz=10.0
)

# ================================
# 3. CALCULATE PERFORMANCE METRICS
# ================================
true_latencies = mock_data['true_latencies_ms']
true_latencies_centered = true_latencies - np.median(true_latencies)

# Woody's metrics
woody_correlation = np.corrcoef(true_latencies_centered, woody_results['latencies_ms'])[0, 1]
woody_rmse = np.sqrt(np.mean((true_latencies_centered - woody_results['latencies_ms'])**2))

# RIDE metrics (use first component only)
ride_latencies = ride_results['latencies_ms'][:, 0]  # First (and only) component
ride_correlation = np.corrcoef(true_latencies_centered, ride_latencies)[0, 1]
ride_rmse = np.sqrt(np.mean((true_latencies_centered - ride_latencies)**2))

print(f"\nPERFORMANCE COMPARISON:")
print(f"Woody's Algorithm - Correlation: {woody_correlation:.3f}, RMSE: {woody_rmse:.2f} ms")
print(f"RIDE Algorithm    - Correlation: {ride_correlation:.3f}, RMSE: {ride_rmse:.2f} ms")

# ================================
# 4. CREATE COMPARISON PLOTS
# ================================
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
n_trials_to_plot = min(10, X.shape[0])

# ================================
# TOP ROW: INDIVIDUAL TRIALS
# ================================

# Plot 1: Original individual trials
for i in range(n_trials_to_plot):
    axes[0, 0].plot(time_axis, X[i, :], alpha=0.4, color='gray', linewidth=0.8)
axes[0, 0].plot(time_axis, mock_data['template_erp'], 'k--', label='True Template', linewidth=2)
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude (μV)')
axes[0, 0].set_title(f'Original Trials (first {n_trials_to_plot})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Woody's aligned trials
woody_aligned_data = np.zeros_like(X)
for trial_idx in range(X.shape[0]):
    woody_aligned_data[trial_idx, :] = np.roll(X[trial_idx, :], -int(woody_results['latencies_ms'][trial_idx] * sfreq / 1000))

for i in range(n_trials_to_plot):
    axes[0, 1].plot(time_axis, woody_aligned_data[i, :], alpha=0.4, color='blue', linewidth=0.8)
axes[0, 1].plot(time_axis, woody_results['aligned_erp'], 'r-', label="Woody's Aligned Avg", linewidth=2)
axes[0, 1].plot(time_axis, mock_data['template_erp'], 'k--', label='True Template', linewidth=2)
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Amplitude (μV)')
axes[0, 1].set_title(f"Woody's Aligned Trials")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: RIDE aligned trials
ride_aligned_data = np.zeros_like(X)
for trial_idx in range(X.shape[0]):
    ride_aligned_data[trial_idx, :] = np.roll(X[trial_idx, :], -int(ride_latencies[trial_idx] * sfreq / 1000))

for i in range(n_trials_to_plot):
    axes[0, 2].plot(time_axis, ride_aligned_data[i, :], alpha=0.4, color='green', linewidth=0.8)
axes[0, 2].plot(time_axis, ride_results['reconstructed_erp'], 'orange', label='RIDE Reconstructed', linewidth=2)
axes[0, 2].plot(time_axis, mock_data['template_erp'], 'k--', label='True Template', linewidth=2)
axes[0, 2].set_xlabel('Time (ms)')
axes[0, 2].set_ylabel('Amplitude (μV)')
axes[0, 2].set_title('RIDE Aligned Trials')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# ================================
# MIDDLE ROW: ERP COMPARISON
# ================================

# Plot 4: ERP waveforms comparison
axes[1, 0].plot(time_axis, woody_results['original_erp'], 'gray', label='Original ERP', alpha=0.7, linewidth=2)
axes[1, 0].plot(time_axis, woody_results['aligned_erp'], 'blue', label="Woody's Aligned", linewidth=2)
axes[1, 0].plot(time_axis, ride_results['reconstructed_erp'], 'orange', label='RIDE Reconstructed', linewidth=2)
axes[1, 0].plot(time_axis, mock_data['template_erp'], 'k--', label='True Template', linewidth=2)
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Amplitude (μV)')
axes[1, 0].set_title('ERP Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Latency scatter plots
axes[1, 1].scatter(true_latencies_centered, woody_results['latencies_ms'], alpha=0.6, label='Woody', color='blue')
axes[1, 1].scatter(true_latencies_centered, ride_latencies, alpha=0.6, label='RIDE', color='orange')
range_lim = max(np.abs(true_latencies_centered).max(), np.abs(woody_results['latencies_ms']).max(), np.abs(ride_latencies).max()) * 1.1
axes[1, 1].plot([-range_lim, range_lim], [-range_lim, range_lim], 'r--', label='Perfect correlation')
axes[1, 1].set_xlabel('True Latencies (ms)')
axes[1, 1].set_ylabel('Estimated Latencies (ms)')
axes[1, 1].set_title('Latency Recovery Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Error comparison
woody_residuals = woody_results['latencies_ms'] - true_latencies_centered
ride_residuals = ride_latencies - true_latencies_centered
axes[1, 2].scatter(true_latencies_centered, woody_residuals, alpha=0.6, label=f"Woody (RMSE={woody_rmse:.1f})", color='blue')
axes[1, 2].scatter(true_latencies_centered, ride_residuals, alpha=0.6, label=f"RIDE (RMSE={ride_rmse:.1f})", color='orange')
axes[1, 2].axhline(y=0, color='r', linestyle='--')
axes[1, 2].set_xlabel('True Latencies (ms)')
axes[1, 2].set_ylabel('Residuals (ms)')
axes[1, 2].set_title('Error Comparison')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./figures/woody_vs_ride_comparison.png', dpi=300)

# ================================
# PRINT ADDITIONAL METRICS
# ================================
print("\n" + "="*60)
print("DETAILED METRICS")
print("="*60)

print("\nLatency Distribution Statistics:")
print(f"True latencies - Mean: {np.mean(true_latencies):.2f} ms, Std: {np.std(true_latencies):.2f} ms")
print(f"Woody estimates - Mean: {np.mean(woody_results['latencies_ms']):.2f} ms, Std: {np.std(woody_results['latencies_ms']):.2f} ms")
print(f"RIDE estimates - Mean: {np.mean(ride_latencies):.2f} ms, Std: {np.std(ride_latencies):.2f} ms")

print("\nPerformance Metrics Summary:")
print(f"{'Method':<15} {'Correlation':<15} {'RMSE (ms)':<15}")
print("-" * 45)
print(f"{'Woody':<15} {woody_correlation:<15.3f} {woody_rmse:<15.2f}")
print(f"{'RIDE':<15} {ride_correlation:<15.3f} {ride_rmse:<15.2f}")

print("\nComponent Information:")
print(f"RIDE component peak amplitude: {np.max(np.abs(ride_results['components'][:, 0])):.2f} μV")
print(f"True template peak amplitude: {np.max(np.abs(mock_data['template_erp'])):.2f} μV")
#%%