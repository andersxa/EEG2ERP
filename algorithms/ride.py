import numpy as np
from scipy.signal import butter, filtfilt, detrend, correlate

def ride_decomposition(
    X: np.ndarray,
    sfreq: float,
    chosen_channel: int,
    high_cutoff_hz: float = 30.0
):
    """
    Barebones Python implementation of the Residue Iteration Decomposition (RIDE) algorithm.

    This function decomposes single-channel EEG data into three components with
    variable single-trial latencies. It is designed to separate overlapping
    neural components in Event-Related Potential (ERP) analysis.

    Args:
        X (np.ndarray): The input EEG data as a 3D NumPy array with
                        shape (K_trials, C_channels, T_samples).
        sfreq (float): The sampling frequency of the data in Hz.
        chosen_channel (int): The index of the channel to perform the
                              decomposition on.
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
    if X.ndim != 3:
        raise ValueError("Input data `X` must be a 3D array of shape (trials, channels, samples).")

    # Select the chosen channel and transpose to (samples, trials) for easier processing
    data = X[:, chosen_channel, :].T
    n_samples, n_trials = data.shape
    original_erp = np.mean(data, axis=1)

    # --- Configuration (derived from inputs and assumptions) ---
    epoch_start_ms, epoch_end_ms = -200, 800
    comp_names = ['S', 'C', 'R']
    n_components = len(comp_names)

    # Define component time windows in milliseconds relative to the epoch
    # These are reasonable defaults for an S-C-R paradigm
    twd_ms = {
        'S': [50, 200],
        'C': [250, 500],
        'R': [500, 750]
    }

    # Convert ms to sample indices
    def ms_to_samples(ms):
        return int((ms - epoch_start_ms) * sfreq / 1000)

    twd_samp = {name: [ms_to_samples(t[0]), ms_to_samples(t[1])] for name, t in twd_ms.items()}
    dur_samp = {name: (t[1] - t[0]) // 2 for name, t in twd_samp.items()}

    # Initialize filter coefficients
    filter_b, filter_a = _design_filter(sfreq, high_cutoff_hz)

    # ----------------------------------------------------------------------
    # 3. Initial Latency Estimation (Woody's Method)
    # ----------------------------------------------------------------------
    latencies = np.zeros((n_trials, n_components), dtype=int)
    initial_template = original_erp.copy()

    for i, name in enumerate(comp_names):
        win = slice(twd_samp[name][0], twd_samp[name][1])
        search_dur = dur_samp[name]
        template_segment = initial_template[win]

        for trial_idx in range(n_trials):
            trial_segment = data[win, trial_idx]
            # Cross-correlate trial with the template
            xcorr = correlate(trial_segment - trial_segment.mean(), template_segment - template_segment.mean(), mode='same')
            # Find the peak of the correlation
            peak_idx = np.argmax(xcorr[len(xcorr)//2 - search_dur : len(xcorr)//2 + search_dur])
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

                # The stimulus-locked version is the average of the unaligned component estimates
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
            win = slice(twd_samp[name][0], twd_samp[name][1])
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
                residual_filt[win] = detrend(residual_filt[win])

                # Cross-correlate residual with template
                xcorr = correlate(
                    residual_filt[win] - residual_filt[win].mean(),
                    template_filt[win] - template_filt[win].mean(),
                    mode='same'
                )

                # Find new peak
                center = len(xcorr) // 2
                search_slice = slice(center - search_dur, center + search_dur)
                peak_idx = np.argmax(xcorr[search_slice])
                latencies[trial_idx, i] = (center - search_dur) + peak_idx - center

        # Re-center latencies after each outer iteration
        latencies -= np.median(latencies, axis=0).astype(int)

    # ----------------------------------------------------------------------
    # 5. Finalization and Output
    # ----------------------------------------------------------------------
    reconstructed_erp = np.sum(comp_sl, axis=1)
    
    # Convert latencies from samples to milliseconds for the output
    latencies_ms = latencies * (1000.0 / sfreq)

    return {
        'reconstructed_erp': reconstructed_erp,
        'original_erp': original_erp,
        'components': comp,
        'latencies_ms': latencies_ms
    }