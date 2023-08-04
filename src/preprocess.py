import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer



CAT_FEATURES = [
    "season",
    "weather"
]



preprocess_pipeline = ColumnTransformer(
    transformers=[
        ("target_encoding", TargetEncoder(), CAT_FEATURES)
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")