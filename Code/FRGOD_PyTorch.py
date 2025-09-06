import numpy as np
import torch  # Import PyTorch
import os

# -------------------------- 1. Auto-detect GPU/CPU device --------------------------
# Prioritize using GPU (cuda:0), use CPU if no GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_diagnosis = []

# Generate device diagnosis information
if torch.cuda.is_available():
    gpu_diagnosis.append(f"✅ Detected available GPU: {torch.cuda.get_device_name(0)}")
    gpu_diagnosis.append(f"✅ Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    gpu_diagnosis.append(f"✅ PyTorch version: {torch.__version__}")
else:
    gpu_diagnosis.append("❌ No available GPU detected, will use CPU")
    gpu_diagnosis.append(f"✅ PyTorch version: {torch.__version__}")

# Check CUDA environment variable (for troubleshooting)
cuda_path = os.environ.get('CUDA_PATH')
if cuda_path:
    gpu_diagnosis.append(f"ℹ️ CUDA_PATH environment variable: {cuda_path}")
else:
    gpu_diagnosis.append("ℹ️ CUDA_PATH environment variable not set (only affects CUDA toolchain recognition, not PyTorch GPU usage)")

# -------------------------- 2. Import dependent libraries --------------------------
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

# -------------------------- 3. FRGOD core function (PyTorch implementation) --------------------------
def FRGOD(data, lammda):
    ## Detecting Fuzzy Rough Granules-based Outlier Detection (FRGOD) algorithm
    ## input:
    # data: data matrix (rows = samples, columns = attributes, no decision column)
    # lammda: adaptive fuzzy radius adjustment parameter
    ## output:
    # FRGOF: Fuzzy Rough Granule Outlier Factor (NumPy array, outlier score for each sample)

    # Step 1: Data conversion (NumPy array → PyTorch tensor, move to target device)
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    print(f"ℹ️ Data loaded to {device}, data shape: {data.shape}")

    n, m = data.shape

    # Step 2: Calculate adaptive fuzzy radius delta
    delta = torch.zeros(m, dtype=torch.float32, device=device)
    is_valid = (data <= 1).all(dim=0) & (data.max(dim=0)[0] != data.min(dim=0)[0])
    if is_valid.any():
        delta[is_valid] = data[:, is_valid].std(dim=0) / lammda
    print("ℹ️ Adaptive fuzzy radius delta calculation completed")

    # Step 3: Construct neighbor set NbrSet (n×n×m, neighbor similarity for each attribute)
    NbrSet = torch.zeros((n, n, m), dtype=torch.float32, device=device)
    for col in range(m):
        col_data = data[:, col].reshape(-1, 1)

        dist = torch.abs(col_data - col_data.T)

        dist[dist > delta[col]] = 1.0
        similarity = 1.0 - dist
        NbrSet[:, :, col] = similarity
    print("ℹ️ Neighbor set NbrSet construction completed")

    # Step 4: Calculate attribute importance and accuracy (Acc_A_a)
    LA = torch.arange(m, device=device)
    weight1 = torch.zeros((n, m), dtype=torch.float32, device=device)
    weight3 = torch.zeros((n, m), dtype=torch.float32, device=device)
    Acc_A_a = torch.zeros((n, m, m), dtype=torch.float32, device=device)

    for col in range(m):
        lA_d = torch.tensor([x for x in LA if x != col], device=device)
        Acc_A_a_tem = torch.zeros((n, m), dtype=torch.float32, device=device)
        NbrSet_tem = NbrSet[:, :, col].clone()

        NbrSet_tem_np = NbrSet_tem.cpu().numpy()
        unique_patterns_np, ic_np = np.unique(NbrSet_tem_np, axis=0, return_inverse=True)
        unique_patterns = torch.as_tensor(unique_patterns_np, dtype=torch.float32, device=device)
        ic = torch.as_tensor(ic_np, dtype=torch.int64, device=device)

        for i in range(unique_patterns.shape[0]):
            RM_temp = unique_patterns[i].unsqueeze(0).repeat(n, 1)
            sample_indices = torch.where(ic == i)[0]

            for k in range(-1, m - 1):
                if k == -1:
                    lA_de = lA_d
                else:
                    lA_de = torch.tensor([x for x in lA_d if x != lA_d[k]], device=device)

                NbrSet_tmp = NbrSet[:, :, lA_de].min(dim=2)[0]

                low_term = torch.maximum(1.0 - NbrSet_tmp, RM_temp).min(dim=1)[0]
                Low_A = low_term.sum()

                upp_term = torch.minimum(NbrSet_tmp, RM_temp).max(dim=1)[0]
                Upp_A = upp_term.sum()

                if Upp_A < 1e-8:
                    Acc_A_a_tem[sample_indices, k + 1] = 0.0
                else:
                    Acc_A_a_tem[sample_indices, k + 1] = Low_A / Upp_A

            Acc_A_a[:, :, col] = Acc_A_a_tem
            avg_similarity = unique_patterns[i].mean()
            weight3[sample_indices, col] = 1.0 - (avg_similarity) ** (1 / 3)
            weight1[sample_indices, col] = avg_similarity
    print("ℹ️ Attribute accuracy and weight calculation completed")

    # Step 5: Calculate FSDOG (Fuzzy Rough Granule Dissimilarity)
    FSDOG = torch.zeros((n, m), dtype=torch.float32, device=device)
    for col in range(m):
        Acc_A_a_tem = Acc_A_a[:, :, col]
        sum_term = ((Acc_A_a_tem[:, 1:m] + 1.0) / 2.0).sum(dim=1) + Acc_A_a_tem[:, 0]
        FSDOG[:, col] = 1.0 - (sum_term / m) * weight1[:, col]
    print("ℹ️ FSDOG calculation completed")

    # Step 6: Calculate final outlier factor FRGOF (average along attribute dimension)
    FRGOF = (FSDOG * weight3).mean(dim=1)

    # Step 7: Convert back to NumPy array (facilitate subsequent analysis, consistent with original code output format)
    FRGOF_np = FRGOF.cpu().numpy()
    print(f"ℹ️ Calculation completed, results transferred from {device} back to CPU (NumPy array)")

    return FRGOF_np


# -------------------------- 4. Main function (test entry) --------------------------
if __name__ == "__main__":
    # Print device diagnosis information
    print("=" * 60)
    print("PyTorch GPU environment diagnosis report:")
    for msg in gpu_diagnosis:
        print(f"- {msg}")
    print("=" * 60)
    print(f"Final computing device: {device}")
    print()

    # Load data (MATLAB format data)
    load_data = loadmat('Example.mat')
    trandata = load_data['Example']
    print(f"Loaded data shape: {trandata.shape}")

    # Normalize numerical attributes
    scaler = MinMaxScaler()
    trandata[:, 1:3] = scaler.fit_transform(trandata[:, 1:3])
    print("Numerical attribute normalization completed")

    # Run FRGOD algorithm
    lam = 1.0
    out_factor = FRGOD(trandata, lam)

    # Output results
    print("\n" + "=" * 60)
    print("FRGOD outlier factor results (outlier score for each sample):")
    print(out_factor)
    print("=" * 60)
