import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import glob
import torch
from tqdm import tqdm

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ---------------------------
# Helpers
# ---------------------------
def to_4x4(T_3x4: torch.Tensor) -> torch.Tensor:
    """(S,3,4) -> (S,4,4)"""
    assert T_3x4.ndim == 3 and T_3x4.shape[-2:] == (3, 4)
    S = T_3x4.shape[0]
    T = torch.eye(4, device=T_3x4.device, dtype=T_3x4.dtype).unsqueeze(0).repeat(S, 1, 1)
    T[:, :3, :] = T_3x4
    return T


def write_kitti_poses(T_0i: torch.Tensor, path: str) -> None:
    """
    KITTI odometry pose format:
      each line = 12 floats for a 3x4 matrix
      this matrix maps a point from camera-i coords into camera-0 coords
      i.e., T_{c0 <- ci}
    """
    T = T_0i.detach().cpu().numpy()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for i in range(T.shape[0]):
            f.write(" ".join(f"{v:.10f}" for v in T[i, :3, :].reshape(-1)) + "\n")


# ---------------------------
# VGGT inference (chunk)
# ---------------------------
@torch.inference_mode()
def infer_chunk_world_to_cam(model: VGGT, images_SCHW: torch.Tensor, amp_dtype: torch.dtype):
    """
    images_SCHW: (S,3,H,W) on device
    returns:
      T_cw: (S,4,4) world->cam  (OpenCV "camera from world")
      K:    (S,3,3)
    """
    images = images_SCHW.unsqueeze(0)  # (1,S,3,H,W)

    use_amp = images.is_cuda and amp_dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else torch.autocast("cpu", enabled=False)

    with autocast_ctx:
        toks, _ps_idx = model.aggregator(images)
        pose_enc = model.camera_head(toks)[-1]
        extr_3x4, K = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    T_cw = to_4x4(extr_3x4.squeeze(0).float())  # (S,4,4)
    K = K.squeeze(0).float()                    # (S,3,3)
    return T_cw, K


# ---------------------------
# Stitching into KITTI poses
# ---------------------------
@torch.inference_mode()
def stitch_kitti_poses_chunked(
    model: VGGT,
    image_paths: list[str],
    device: str,
    amp_dtype: torch.dtype,
    chunk: int = 100,
    overlap: int = 1,
):
    """
    Returns:
      T_0i_out: (N,4,4) on CPU, KITTI convention: maps cam_i -> cam_0
    """
    assert chunk >= 2
    assert overlap >= 1
    N = len(image_paths)
    step = chunk - overlap
    assert step >= 1, "Need chunk > overlap"

    T_0i_out = [None] * N
    committed = {}  # global_idx -> (4,4) CPU

    for start in tqdm(range(0, N, step)):
        end = min(start + chunk, N)
        S = end - start

        # if this chunk only contains frames already covered by overlap, skip
        if start > 0 and S <= overlap:
            continue

        # load images for this chunk
        imgs = load_and_preprocess_images(image_paths[start:end]).to(device)  # (S,3,H,W)

        # VGGT extrinsics: world->cam, per-chunk world gauge
        T_cw, _K = infer_chunk_world_to_cam(model, imgs, amp_dtype)

        # invert to cam->world_local
        T_wc = torch.linalg.inv(T_cw)  # (S,4,4)

        if start == 0:
            # define cam0 as global reference:
            # T_0i = inv(T_w0) @ T_wi  => cam_i -> cam0
            T_0i = torch.linalg.inv(T_wc[0]) @ T_wc
        else:
            # overlap frame is global index == start, local index 0
            idx_ov = start
            if idx_ov not in committed:
                raise RuntimeError(
                    f"Missing committed overlap pose for frame {idx_ov}. "
                    f"Increase overlap or check chunk/step logic."
                )

            T_0ov = committed[idx_ov].to(device)  # cam_ov -> cam0 (global)
            T_wov = T_wc[0]                       # cam_ov -> world_local (this chunk)

            # Find A: world_local -> cam0 such that T_0ov == A @ T_wov
            # => A = T_0ov @ inv(T_wov)
            A = T_0ov @ torch.linalg.inv(T_wov)

            # apply to all frames in this chunk
            T_0i = A @ T_wc
            T_0i[0] = T_0ov  # enforce exact overlap match

        # commit (skip duplicate overlap frames)
        k0 = 0 if start == 0 else overlap
        for k in range(k0, S):
            g = start + k
            T_0i_out[g] = T_0i[k].detach().cpu()
            committed[g] = T_0i_out[g]

        del imgs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # print(f"[chunk] {start:6d}..{end:6d}  committed={S - k0:4d}")

    # stack and rebase so frame0 is exactly identity
    T_0i_out = torch.stack(T_0i_out, dim=0)  # (N,4,4) CPU
    T_0i_out = torch.linalg.inv(T_0i_out[0]) @ T_0i_out
    return T_0i_out


# ---------------------------
# Optional: abs <-> rel sanity
# ---------------------------
def rel_from_abs(T_0i: torch.Tensor) -> torch.Tensor:
    """rel[i] = T_{ci <- c(i+1)} = inv(T_0i[i]) @ T_0i[i+1]"""
    return torch.linalg.inv(T_0i[:-1]) @ T_0i[1:]


def integrate_rel(rel_i_ip1: torch.Tensor) -> torch.Tensor:
    """T_0,0=I ; T_0,i+1 = T_0,i @ rel[i]"""
    assert rel_i_ip1.ndim == 3 and rel_i_ip1.shape[-2:] == (4, 4)
    device = rel_i_ip1.device
    dtype = rel_i_ip1.dtype

    N = rel_i_ip1.shape[0] + 1
    T_0i = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
    for i in range(N - 1):
        T_0i[i + 1] = T_0i[i] @ rel_i_ip1[i]
    return T_0i


# ---------------------------
# Main
# ---------------------------
def main():
    # ---- config ----
    seq_list = sorted(os.listdir("/data_ssd/lei/KITTI_dataset/dataset/sequences/"))[:11]
    # seq_list = sorted(os.listdir("/data/lei/DeepVO/nuScenes_dataset/NUSC_12hz/CAM_FRONT/sequences/"))[:200]
    # seq_list = sorted(os.listdir("/data/lei/DeepVO/nuScenes_dataset/NUSC_12hz/CAM_FRONT/sequences/"))[200:400]
    # seq_list = sorted(os.listdir("/data/lei/DeepVO/nuScenes_dataset/NUSC_12hz/CAM_FRONT/sequences/"))[400:600]
    # seq_list = sorted(os.listdir("/data/lei/DeepVO/nuScenes_dataset/NUSC_12hz/CAM_FRONT/sequences/"))[600:]

    chunk = 150
    overlap = 1

    # ---- device / amp ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major = torch.cuda.get_device_capability()[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        amp_dtype = torch.float32  # amp disabled on cpu in this script

    # ---- load model ----
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    for seq in seq_list:
        seq_dir = f"/data_ssd/lei/KITTI_dataset/dataset/sequences/{seq}/image_2"
        out_path = f"KITTI/{seq}.txt"
        # seq_dir = f"/data/lei/DeepVO/nuScenes_dataset/NUSC_12hz/CAM_FRONT/sequences/{seq}/image_2"
        # out_path = f"NUSC/{seq}.txt"
            
        # ---- image list ----
        image_paths = sorted(glob.glob(os.path.join(seq_dir, "*")))
        if not image_paths:
            raise ValueError(f"No images found in: {seq_dir}")

        print(f"Found {len(image_paths)} images")
        print(f"device={device}, amp_dtype={amp_dtype}, chunk={chunk}, overlap={overlap}")

        # ---- stitch poses ----
        T_0i = stitch_kitti_poses_chunked(
            model=model,
            image_paths=image_paths,
            device=device,
            amp_dtype=amp_dtype,
            chunk=chunk,
            overlap=overlap,
        )

        print("T_0i:", tuple(T_0i.shape), "dtype:", T_0i.dtype, "device:", T_0i.device)

        # ---- write KITTI pose file ----
        write_kitti_poses(T_0i, out_path)
        print("Wrote:", out_path)

        # ---- sanity check (abs -> rel -> abs) ----
        rel = rel_from_abs(T_0i)
        T_0i_rec = integrate_rel(rel)
        max_err = (T_0i_rec - T_0i).abs().max().item()
        print(f"[sanity] max|recovered - original| = {max_err:.3e}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
