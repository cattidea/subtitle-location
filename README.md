# subtitle-location

字幕定位与识别

- 定位 采用 CNN ，模型参考 vgg-16 ，数据集可通过百度图片等方式进行获取，打标签可使用[pic_labeler](https://github.com/SigureMo/notev/tree/master/Codes/VB/08_pic_labeler)，图片与标签的对应关系可参考[deeplearning.ai 3.3.1](https://blog.csdn.net/u013555719/article/details/81637228)
- 识别 暂时使用了百度 OCR 接口

## Usage

- 训练

   ``` python
   python main.py train
   ```

   - 参数 `--resume` : 恢复模型继续训练

- 测试视频文件

   ``` python
   python main.py test
   ```

   - ~~参数 `--use-cache` : 使用 cache~~
