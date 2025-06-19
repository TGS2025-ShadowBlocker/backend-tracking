import cv2

class WebCamCapture:
    def __init__(self):
        # ウェブカメラのキャプチャを初期化
        self.cap = cv2.VideoCapture(0)

    def start_capture(self):
        """
        ウェブカメラのキャプチャを開始
        """
        print("'q'キーで終了")
        # キャプチャがオープンしている間続ける
        while(self.cap.isOpened()):
            # フレームを読み込む
            ret, frame = self.cap.read()
            if ret == True:
                # 左右反転
                frame = cv2.flip(frame, 1)

                # フレームを表示
                cv2.imshow('Webcam Live', frame)

                # 'q'キーが押されたらループから抜ける
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # キャプチャをリリースし、ウィンドウを閉じる
        self.cleanup()

    def cleanup(self):
        """
        リソースのクリーンアップ
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        """
        デストラクタでリソースをクリーンアップ
        """
        if hasattr(self, 'cap'):
            self.cleanup()


# 使用例
if __name__ == "__main__":
    webcam = WebCamCapture()
    webcam.start_capture()
