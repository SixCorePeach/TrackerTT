import argparse
from tqdm import tqdm
import os


def main(dir,done_delete):
    if(os.path.exists(dir)):
        #zip_dir 列出dir里所有文件
        for zip_dir in tqdm(os.listdir(dir)):
            if '.zip' in zip_dir:         
                    cmd_str_cd = f"cd {dir}"  
                    cmd_str_unzip = f'unzip -q {zip_dir}'  
                    cmd_2 = f'{cmd_str_cd};{cmd_str_unzip}'
                    print('正在解压文件...(耐心等待)')
                    return_code = os.system(cmd_2)
                    # 检查命令是否成功运行
                    if return_code == 0:
                        print("unzip命令成功运行")
                    else:
                        print("unzip命令运行失败")
            else:
                #只执行是文件夹的zip_dir
                if os.path.isdir(f'{dir}/{zip_dir}'):
                    print(f'当前正在处理{zip_dir}')
                    file_name = f'{dir}/{zip_dir}/{zip_dir}.zip'
                    
                    # cat
                    if os.path.exists(file_name):
                        print(f'{zip_dir}.zip existes, not need to cat')
                    else:
                        cmd_str_cd = f"cd {dir}/{zip_dir}"  
                        cmd_str_cat = f'cat {zip_dir}.zip* > {zip_dir}.zip'  
                        cmd_1 = f'{cmd_str_cd};{cmd_str_cat}'
                        print(f'正在cat原文件为{zip_dir}.zip...(耐心等待)')
                        return_code = os.system(cmd_1)
                        # 检查命令是否成功运行
                        if return_code == 0:
                            print("cat命令成功运行")
                        else:
                            print("cat命令运行失败")
                    # delete 原文件
                    if done_delete:
                        cmd_str_cd = f"cd {dir}/{zip_dir}"  
                        cmd_str_delete = f'rm *.zip.0*'  
                        cmd_2 = f'{cmd_str_cd};{cmd_str_delete}'
                        print('正在删除原文件...(耐心等待)')
                        return_code = os.system(cmd_2)
                        # 检查命令是否成功运行
                        if return_code == 0:
                            print("rm命令成功运行")
                        else:
                            print("rm命令运行失败")
                            
                    # 整理 （发现多嵌套了一层目录）
                    if (os.path.exists(f"{dir}/{zip_dir}/{zip_dir}.zip")):
                        print("解压后文件多嵌入了一层目录")
                        cmd_mv = f'mv {dir}/{zip_dir}/{zip_dir}.zip {dir}'
                        print('正在mv文件...(耐心等待)')
                        return_code = os.system(cmd_mv)
                        # 检查命令是否成功运行
                        if return_code == 0:
                            print("mv命令成功运行")
                        else:
                            print("mv命令运行失败")
                        
                    # unzip 解压
                    if (os.path.exists(f"{dir}/{zip_dir}.zip")):    
                        cmd_str_cd = f'cd {dir}'
                        cmd_str_rm = f'rm -rf {zip_dir}'
                        cmd_3 = f'{cmd_str_cd};{cmd_str_rm}'
                        print('正在删除文件夹...(耐心等待)')
                        return_code = os.system(cmd_3)
                        # 检查命令是否成功运行
                        if return_code == 0:
                            print("rm命令成功运行")
                        else:
                            print("rm命令运行失败")  
                                
                        cmd_str_cd = f"cd {dir}"  
                        cmd_str_unzip = f'unzip -q {zip_dir}.zip'  
                        cmd_2 = f'{cmd_str_cd};{cmd_str_unzip}'
                        print('正在解压文件...(耐心等待)')
                        return_code = os.system(cmd_2)
                        # 检查命令是否成功运行
                        if return_code == 0:
                            print("unzip命令成功运行")
                        else:
                            print("unzip命令运行失败")
                        
            
    else:
        raise TypeError
        



if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Extract the frames for TrackingNet')
    p.add_argument('--trackingnet_dir', type=str, default='/data3/Track_datasets/trackingnet',help='Main TrackingNet folder.')
    p.add_argument('--done_delete', action='store_false',help='cat done delete origin file.')
    args = p.parse_args()
 
    main(args.trackingnet_dir,args.done_delete)