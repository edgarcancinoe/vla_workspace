import math
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class PhotometricAugment(torch.nn.Module):
    """
    (1) ColorJitter, (2) Gaussian noise, (3) Gaussian blur
    Expects x: (C,H,W) or (B,C,H,W), float in [0,1]
    """
    def __init__(
        self,
        enable=True,
        brightness=0.30,
        contrast=0.30,
        saturation=0.20,
        hue=0.05,
        p_color=0.9,
        noise_std=0.01,
        p_noise=0.7,
        blur_kernel=(3, 3),
        blur_sigma=(0.1, 1.2),
        p_blur=0.3,
    ):
        super().__init__()
        self.enable = enable
        self.p_color = float(p_color)
        self.p_noise = float(p_noise)
        self.p_blur = float(p_blur)
        self.noise_std = float(noise_std)

        self.color = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.blur = T.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.enable) or (not torch.is_tensor(x)):
            return x

        # Ensure float32
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        # Batch-wise random checks for performance
        if torch.rand((), device=x.device) < self.p_color:
            x = self.color(x)

        if self.noise_std > 0 and torch.rand((), device=x.device) < self.p_noise:
            x = x + torch.randn_like(x) * self.noise_std

        if torch.rand((), device=x.device) < self.p_blur:
            x = self.blur(x)

        return x.clamp(0.0, 1.0)


class GeometricAugment(torch.nn.Module):
    """
    Random affine with selectable border handling.
    Now supports BATCHED processing for speed.
    Expects x: (C,H,W) or (B,C,H,W), float in [0,1]
    """
    def __init__(
        self,
        enable=True,
        rotation_deg=15.0,
        translation_frac=0.10,
        mode="reflect",             # "reflect" | "crop" | "black"
        interpolation=F.InterpolationMode.BILINEAR,
    ):
        super().__init__()
        assert mode in {"reflect", "crop", "black"}
        self.enable = enable
        self.rotation_deg = float(rotation_deg)
        self.translation_frac = float(translation_frac)
        self.mode = mode
        self.interp = interpolation

    @staticmethod
    def _center_crop(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        # x: (..., C, H, W)
        H, W = x.shape[-2:]
        top = max(0, (H - out_h) // 2)
        left = max(0, (W - out_w) // 2)
        return x[..., top:top + out_h, left:left + out_w]

    @staticmethod
    def _resize(x: torch.Tensor, out_h: int, out_w: int, interp) -> torch.Tensor:
        # F.resize handles (..., C, H, W)
        return F.resize(x, [out_h, out_w], interpolation=interp, antialias=True)

    def _safe_crop_factor(self, theta_rad: float) -> float:
        c = abs(math.cos(theta_rad))
        s = abs(math.sin(theta_rad))
        f_rot = 1.0 / max(1e-6, (c + s))
        f_trans = max(0.0, 1.0 - 2.0 * self.translation_frac)
        return max(0.0, min(1.0, f_rot, f_trans))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.enable) or (not torch.is_tensor(x)):
            return x

        # Ensure float32
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        # Handle both (C,H,W) and (B,C,H,W)
        is_batched = x.dim() == 4
        if not is_batched:
            # We still use batched logic for consistency
            x = x.unsqueeze(0)
            
        B, C, H, W = x.shape

        # Sample random params (one per batch for consistency or separate? 
        # Typically we want separate for each image in batch for variety)
        # However, F.affine takes single params for the whole batch.
        # To get variety, we can loop (slow) or use a per-batch logic if supported.
        # Torchvision 0.17+ supports batch transforms with v2, but here we use functional.
        
        # For performance, we'll use same params for the batch if we want it FAST, 
        # or we can iterate if we want VARIETY. 
        # Actually, in Finetuning, variety within batch is critical.
        
        # Let's check torchvision version
        import torchvision
        from packaging import version
        tv_version = version.parse(torchvision.__version__)
        
        # We will loop for now as it's still 100x faster on GPU than CPU-DataLoader-with-shm
        # But even better: if we are in "train", we can just iterate.
        
        y_list = []
        for i in range(B):
            xi = x[i]
            theta = (torch.rand((), device=x.device).item() * 2 - 1) * self.rotation_deg
            max_dx = self.translation_frac * W
            max_dy = self.translation_frac * H
            dx = (torch.rand((), device=x.device).item() * 2 - 1) * max_dx
            dy = (torch.rand((), device=x.device).item() * 2 - 1) * max_dy
            theta_rad = math.radians(theta)

            if self.mode == "reflect":
                diag = math.sqrt(H*H + W*W)
                rot_slack = 0.5 * max(0.0, diag - min(H, W))
                pad = int(math.ceil(max(max_dx, max_dy) + rot_slack))
                
                xi_pad = torch.nn.functional.pad(xi, (pad, pad, pad, pad), mode="reflect")
                yi = F.affine(xi_pad, angle=theta, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0], interpolation=self.interp)
                yi = self._center_crop(yi, H, W)
                y_list.append(yi)
            
            elif self.mode == "crop":
                yi = F.affine(xi, angle=theta, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0], interpolation=self.interp)
                f = self._safe_crop_factor(theta_rad)
                out_h, out_w = max(2, int(round(H*f))), max(2, int(round(W*f)))
                yi = self._center_crop(yi, out_h, out_w)
                yi = self._resize(yi.unsqueeze(0), H, W, self.interp).squeeze(0)
                y_list.append(yi)
            
            else: # black
                yi = F.affine(xi, angle=theta, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0], interpolation=self.interp, fill=0)
                y_list.append(yi)

        y = torch.stack(y_list).clamp(0.0, 1.0)
        return y if is_batched else y.squeeze(0)


class CustomAugmentationPipeline(torch.nn.Module):
    """
    Combines Geometric and Photometric augmentations.
    Optimized for GPU batch processing.
    """
    def __init__(
        self,
        enable_geometric=True,
        rotation_deg=15.0,
        translation_frac=0.10,
        fill_mode="reflect",
        enable_photometric=True,
    ):
        super().__init__()
        self.geometric = GeometricAugment(
            enable=enable_geometric,
            rotation_deg=rotation_deg,
            translation_frac=translation_frac,
            mode=fill_mode,
        )
        self.photometric = PhotometricAugment(enable=enable_photometric)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be (C,H,W) or (B,C,H,W)
        x = self.geometric(x)
        x = self.photometric(x)
        return x
