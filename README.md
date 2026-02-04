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
3) 클립 임베딩으로 K-means 군집화 + 중심 근접도 점수 산출  
4) 군집별 후보에서 프레임 변화 점수 계산 (CLIP ViT)  
5) 점수 결합: `a*C + b*D + c*(C*D)`  
6) 군집별 대표 클립 선택 → 요약 영상 생성  

---

## 사용법

### 1) 의존성 설치
```
uv pip install -r requirements.txt
```

### 2) 입력 파일 경로 수정
`main.py`에서 `input_path`를 원하는 파일로 바꿉니다.

```python
# main.py
input_path = Path("data/origin/call_of_duty.mp4")
```

### 3) 실행
```
uv run main.py
```

### 4) 결과
- 요약 영상: `data/summary_videos/` 폴더에 저장

---

## 설정값 변경
모든 주요 파라미터는 `config.py`에서 관리합니다.  
모델 설정(모델 ID, 디바이스 등)도 `config.py`에서 수정할 수 있습니다.

예: 요약 길이 비율 조정
```python
# main.py
from config import PipelineConfig, ClusterScoreConfig

pipe_cfg = PipelineConfig(cluster=ClusterScoreConfig(k_ratio=0.1))
```

---

## 참고
- `torchvision.io.read_video`는 PyAV가 필요합니다.
- PE-AV 입력 오디오는 48kHz로 맞춰집니다.
