import imageio
import os
gif = []
# 存放多张测试图片的路径拼接

dir_path = r'D:\myproject\CODE\Goprodataset\val\ave_bicubic'
# 获取该文件夹内的全部文件
png = os.listdir(dir_path)
num=0
for i in png:
	num+=1
	if num>100:
		break
	
	if int(i[-5])%2==0:
	# 添加图片，传入参数为图片地址，需拼接路径
	    gif.append(imageio.imread(os.path.join(dir_path, i)))
    

# 生成GIF图
imageio.mimsave("test2.gif", gif, fps=5)	# fps值越大，生成的gif图播放就越快
