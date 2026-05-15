import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_from_excel():
    # 현재 스크립트 위치 기준 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, "episode_rewards.xlsx")
    image_path = os.path.join(current_dir, "scores.png")

    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} 파일을 찾을 수 없습니다.")
        return

    # 엑셀 파일 로드
    df = pd.read_excel(excel_path)
    if 'score' not in df.columns:
        print("Error: 엑셀 파일에 'score' 컬럼이 없습니다.")
        return

    scores = df['score'].toli근st()

    # 그래프 그리기
    plt.figure(figsize=(10, 5))
    
    # Raw Reward (연한 파란색)
    plt.plot(scores, alpha=0.3, color='blue', label='Raw Reward')
    
    # Smoothed Reward (주황색 이동 평균)
    if len(scores) >= 10:
        smooth_scores = pd.Series(scores).rolling(window=10).mean()
        plt.plot(smooth_scores, color='orange', linewidth=2, label='Smoothed Reward (MA 10)')
    
    plt.title('DQN Training Scores (from Excel)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 저장 및 출력
    plt.savefig(image_path, dpi=300)
    print(f"그래프가 성공적으로 저장되었습니다: {image_path}")
    plt.show()

if __name__ == "__main__":
    plot_from_excel()
