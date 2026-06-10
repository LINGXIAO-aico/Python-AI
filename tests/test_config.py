"""配置模块测试。"""

from pathlib import Path

import campus_rag.config as cfg


class TestConfig:
    def test_project_root_exists(self):
        assert cfg.PROJECT_ROOT.exists()
        assert cfg.PROJECT_ROOT.is_dir()

    def test_data_dirs_defined(self):
        assert cfg.RAW_DIR.name == "raw"
        assert cfg.PROCESSED_DIR.name == "processed"

    def test_model_paths(self):
        assert isinstance(cfg.INDEX_PATH, Path)
        assert isinstance(cfg.FAISS_INDEX_PATH, Path)
        assert isinstance(cfg.BGE_CACHE_DIR, Path)

    def test_log_paths(self):
        assert isinstance(cfg.EVAL_SUMMARY_PATH, Path)
        assert isinstance(cfg.ABLATION_RESULTS_PATH, Path)

    def test_deepseek_config(self):
        assert isinstance(cfg.DEEPSEEK_BASE_URL, str)
        assert "deepseek" in cfg.DEEPSEEK_BASE_URL
        assert cfg.DEEPSEEK_CHAT_MODEL == "deepseek-chat"

    def test_embedding_config(self):
        assert "bge" in cfg.EMBEDDING_MODEL_NAME.lower()
        assert cfg.EMBEDDING_DIM == 1024
        assert cfg.EMBEDDING_BATCH_SIZE > 0

    def test_chunk_config(self):
        assert cfg.CHUNK_SIZE == 360
        assert cfg.CHUNK_OVERLAP == 80
