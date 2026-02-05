# Video Summary Pipeline

## 목적
긴 영상을 요약(하이라이트)하기 위한 파이프라인입니다.  
라벨 없이 규칙 기반 점수로 클립을 선택하고, 선택된 클립을 이어 붙여 요약 영상을 생성합니다.

---

## 사용 모델
- **Audio/Video encoder**: `facebook/pe-av-base` (PE-AV)
- **Frame-level ViT**: `openai/clip-vit-base-patch32`

> PE-AV는 오디오와 비디오를 동시에 입력받아 임베딩을 생성합니다.

---

## 파이프라인 흐름
1) 입력 영상 → 고정 길이 클립으로 분할 (겹침 포함)  
2) 각 클립 임베딩 추출 (PE-AV)  
3) 클립 임베딩으로 K-means 군집화 후 후보 추출  
   - 클러스터별로 중심 샘플 + 경계 샘플을 함께 선택  
4) 후보 클립에서 프레임 변화 점수 계산 (CLIP ViT)  
5) 전체 클립 임베딩으로 semantic rare 점수 계산  
6) base 점수 계산: `w_dyn * dynamic + w_sem * semantic`  
7) MMR로 최종 선택 (중복 패널티 반영) → 요약 영상 생성  

---

## 사용법

### 1) 의존성 설치
```
pip install -r requirements.txt
```

### 2) 입력 파일 경로 수정
`main.py`에서 `input_path`를 원하는 파일로 바꿉니다.

```python
# main.py
input_path = Path("examples/input_video.mp4")
```

### 3) 실행
```
python main.py
```

### 4) 결과
- 요약 영상: `data/summary_videos/` 폴더에 저장

---

## 설정값 변경
모든 주요 파라미터는 `config.py`에서 관리합니다.  
모델 설정(모델 ID, 디바이스 등)도 `config.py`에서 수정할 수 있습니다.

예: 군집 후보/최종 선택 파라미터 조정
```python
from config import (
    PipelineConfig,
    ClusterScoreConfig,
    SelectionConfig,
    SemanticScoreConfig,
)

pipe_cfg = PipelineConfig(
    cluster=ClusterScoreConfig(
        k_ratio=0.1,
        top_ratio_per_cluster=0.2,
        boundary_ratio_in_selected=0.4,
    ),
    semantic=SemanticScoreConfig(
        use_rare_score=True,
    ),
    selection=SelectionConfig(
        summary_ratio=0.1,
        dynamic_weight=0.5,
        semantic_weight=0.5,
        lambda_mmr=0.7,
        min_gap_sec=0.0,
    ),
)
```

주요 설정 설명
- `ClusterScoreConfig`
  - `k_ratio`: 전체 클립 대비 클러스터 수 비율
  - `top_ratio_per_cluster`: 클러스터별 후보 선택 비율
  - `boundary_ratio_in_selected`: 후보 중 경계 샘플 비율(나머지는 중심)
- `SelectionConfig`
  - `summary_ratio`: 최종 요약에 포함할 클립 비율
  - `dynamic_weight`: frame-change 점수 가중치
  - `semantic_weight`: semantic rare 점수 가중치
  - `lambda_mmr`: MMR에서 relevance/base 비중 (낮을수록 다양성 강화)
  - `min_gap_sec`: 이미 선택된 클립과 시작시각 최소 간격
- `SemanticScoreConfig`
  - `use_rare_score`: 전체 평균 임베딩 대비 희소성 점수 사용 여부

---

## 참고
- `torchvision.io.read_video`는 PyAV가 필요합니다.
- PE-AV 입력 오디오는 48kHz로 맞춰집니다.
