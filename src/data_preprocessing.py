import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
 
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(_BASE_DIR, "data", "processed")
 
RAW_FILENAME = "data_set_ALL_AML_independent.csv"
 
# Đọc dữ liệu 
def load_data(filepath: str | None = None) -> pd.DataFrame:
    if filepath is None:
        filepath = os.path.join(RAW_DIR, RAW_FILENAME)
 
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {filepath}")
 
    logger.info("Đang đọc dữ liệu từ: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Đọc xong – kích thước: %s hàng × %s cột", *df.shape)
    return df
 
# Lấy ma trận biểu hiện gen thuần tuý 
def _get_expression_matrix(df: pd.DataFrame) -> pd.DataFrame:
    
    # Xác định cột 'call' (tên bắt đầu bằng 'call' sau khi pandas tự thêm '.1', '.2'...)
    call_cols = [c for c in df.columns if str(c).startswith("call")]
    meta_cols = ["Gene Description", "Gene Accession Number"]
    drop_cols = set(call_cols) | set(meta_cols)
 
    expr = df.drop(columns=[c for c in drop_cols if c in df.columns])
 
    # Chuyển tất cả cột sang kiểu số; lỗi chuyển đổi → NaN
    expr = expr.apply(pd.to_numeric, errors="coerce")
 
    # Đặt Gene Accession Number làm index nếu có
    if "Gene Accession Number" in df.columns:
        expr.index = df["Gene Accession Number"].values
        expr.index.name = "Gene Accession Number"
 
    return expr
 
 
# Lọc gen ít biểu hiện / nhiễu
def filter_genes(
    df: pd.DataFrame,
    min_expression: float = 20.0,
    min_fold_change: float = 3.0,
    call_threshold: float = 0.5,
) -> pd.DataFrame:
    logger.info("=== Bắt đầu lọc gen ===")
    n_before = len(df)
 
    # Lấy ma trận biểu hiện thuần tuý
    expr = _get_expression_matrix(df)
 
    # Tiêu chí: Min expression
    max_abs = expr.abs().max(axis=1)
    mask_expr = max_abs >= min_expression
    logger.info(
        "Tiêu chí min_expression (>= %.1f): giữ %d / %d gen",
        min_expression, mask_expr.sum(), n_before
    )
 
    # Tiêu chí: Fold change
    row_max = expr.max(axis=1)
    row_min = expr.min(axis=1)
    # Tránh chia cho 0: nếu min = 0 thì fold = inf (luôn giữ)
    fold_change = (row_max - row_min).abs() / (row_min.abs().replace(0, np.nan))
    fold_change = fold_change.fillna(np.inf)
    mask_fold = fold_change >= min_fold_change
    logger.info(
        "Tiêu chí fold_change (>= %.1f): giữ %d / %d gen",
        min_fold_change, mask_fold.sum(), n_before
    )
 
    # Tiêu chí: Call rate (Absent)
    call_cols = [c for c in df.columns if str(c).startswith("call")]
    if call_cols:
        call_matrix = df[call_cols].values
        absent_rate = (call_matrix == "A").mean(axis=1)
        mask_call = absent_rate < call_threshold
        logger.info(
            "Tiêu chí call_rate (Absent < %.0f%%): giữ %d / %d gen",
            call_threshold * 100, mask_call.sum(), n_before
        )
    else:
        logger.warning("Không tìm thấy cột 'call' – bỏ qua tiêu chí call_rate.")
        mask_call = pd.Series(True, index=df.index)
 
    # Áp dụng tất cả tiêu chí đồng thời
    combined_mask = mask_expr & mask_fold & mask_call
    df_filtered = df[combined_mask].reset_index(drop=True)
 
    logger.info(
        "Kết quả lọc: giữ %d gen (loại bỏ %d gen)",
        len(df_filtered), n_before - len(df_filtered)
    )
    return df_filtered
 
# Xử lý giá trị khuyết thiếu 
def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    valid_strategies = {"median", "mean", "zero", "drop"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"strategy='{strategy}' không hợp lệ. Chọn một trong: {valid_strategies}"
        )
 
    # Xác định cột số để kiểm tra NaN
    expr = _get_expression_matrix(df)
    total_nan = expr.isnull().sum().sum()
 
    if total_nan == 0:
        logger.info("Không có giá trị khuyết thiếu – bỏ qua bước xử lý NaN.")
        return df
 
    logger.info(
        "Phát hiện %d NaN trong ma trận biểu hiện. Áp dụng strategy='%s'.",
        total_nan, strategy
    )
 
    num_cols = expr.columns.tolist()
    df_out = df.copy()
 
    if strategy == "drop":
        # Loại hàng chứa NaN ở bất kỳ cột số nào
        nan_row_mask = df_out[num_cols].isnull().any(axis=1)
        df_out = df_out[~nan_row_mask].reset_index(drop=True)
        logger.info("Đã loại bỏ %d hàng có NaN.", nan_row_mask.sum())
 
    elif strategy == "zero":
        df_out[num_cols] = df_out[num_cols].fillna(0)
 
    elif strategy == "mean":
        row_means = df_out[num_cols].mean(axis=1)
        for col in num_cols:
            df_out[col] = df_out[col].fillna(row_means)
 
    elif strategy == "median":
        row_medians = df_out[num_cols].median(axis=1)
        for col in num_cols:
            df_out[col] = df_out[col].fillna(row_medians)
 
    remaining_nan = df_out[num_cols].isnull().sum().sum()
    logger.info(
        "Sau xử lý: còn %d NaN (đã xử lý %d).",
        remaining_nan, total_nan - remaining_nan
    )
    return df_out
 
# Chuẩn hoá Z-score
def normalize_data(
    df: pd.DataFrame,
    axis: int = 0,
    min_std: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    axis_label = "cột (sample-wise)" if axis == 0 else "hàng (gene-wise)"
    logger.info("=== Chuẩn hoá Z-score theo %s ===", axis_label)
 
    expr = _get_expression_matrix(df)
    num_cols = expr.columns.tolist()
 
    expr_values = expr.values.astype(float)
 
    if axis == 0:
        # Chuẩn hoá theo cột
        mu = expr_values.mean(axis=0, keepdims=True)
        sigma = expr_values.std(axis=0, ddof=1, keepdims=True)
    else:
        # Chuẩn hoá theo hàng
        mu = expr_values.mean(axis=1, keepdims=True)
        sigma = expr_values.std(axis=1, ddof=1, keepdims=True)
 
    # Bảo vệ chia cho 0
    sigma = np.where(sigma < min_std, min_std, sigma)
    z_values = (expr_values - mu) / sigma
 
    # Tạo DataFrame chuẩn hoá (giữ index)
    expr_normalized = pd.DataFrame(
        z_values, index=expr.index, columns=num_cols
    )
 
    # Ghép lại DataFrame đầy đủ (thay cột số bằng giá trị đã chuẩn hoá)
    df_out = df.copy()
    # Dùng vị trí của num_cols trong df để thay thế
    for col in num_cols:
        if col in df_out.columns:
            df_out[col] = expr_normalized[col].values
 
    logger.info(
        "Chuẩn hoá xong – shape ma trận: %d gen × %d mẫu",
        *expr_normalized.shape
    )
    logger.info(
        "Kiểm tra nhanh: mean=%.4f, std=%.4f",
        z_values.mean(), z_values.std()
    )
    return expr_normalized, df_out
 
 
# 6. Lưu dữ liệu đã xử lý 
def save_processed_data(
    expr_normalized: pd.DataFrame,
    filename: str = "gene_expression_normalized.csv",
    output_dir: str | None = None,
) -> str:
    if output_dir is None:
        output_dir = PROCESSED_DIR
 
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
 
    expr_normalized.to_csv(out_path, index=True)
    logger.info("Đã lưu dữ liệu đã xử lý → %s", out_path)
    return out_path
 
# Hàm pipeline tổng hợp (dùng bởi main.py) 
def run_preprocessing(
    filepath: str | None = None,
    filter_kwargs: dict | None = None,
    missing_strategy: str = "median",
    normalize_axis: int = 1,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if filter_kwargs is None:
        filter_kwargs = {}
 
    # Bước 1: Đọc
    df_raw = load_data(filepath)
 
    # Bước 2: Lọc gen
    df_filtered = filter_genes(df_raw, **filter_kwargs)
 
    # Bước 3: Xử lý missing values
    df_clean = handle_missing_values(df_filtered, strategy=missing_strategy)
 
    # Bước 4: Chuẩn hoá
    expr_normalized, df_processed = normalize_data(df_clean, axis=normalize_axis)
 
    # Bước 5: Lưu
    if save:
        save_processed_data(expr_normalized)
 
    return expr_normalized, df_processed
 
