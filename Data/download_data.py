import os
import sys

from ftplib import FTP


class MyFTP(FTP):
    def retrbinary(self, cmd, callback, block_size=0, rest=0):
        """

        :param cmd:
        :param callback:
        :param block_size:
        :param rest:
        :return:
        """

        cmp_size = rest
        self.voidcmd('TYPE I')

        conn = self.transfercmd(cmd, rest)
        while 1:
            if block_size:
                if (block_size - cmp_size) >= 1024:
                    block_size = 1024
                else:
                    block_size = block_size - cmp_size
                ret = float(cmp_size) / block_size
                num = ret * 100

                sys.stdout.write('\r下载进度: {:2f}%'.format(num))
                data = conn.recv(block_size)
                if not data:
                    break
                callback(data)
            cmp_size += block_size
        conn.close()
        return self.voidresp()


def ftp_connect(host: str, username: str, password: str) -> FTP:
    ftp = MyFTP()
    ftp.set_debuglevel(2)
    ftp.connect(host, 21)
    ftp.login(username, password)
    return ftp


# 从ftp下载文件
def download_file(ftp, remote_path, local_path):
    buff_size = 1024
    fp = open(local_path, 'wb')
    ftp.retrbinary(cmd='RETR ' + remote_path,
                   callback=fp.write,
                   block_size=buff_size)
    ftp.set_debuglevel(0)
    fp.close()


# 从本地上传文件到ftp
def upload_file(ftp, remote_path, local_path):
    buff_size = 1024
    fp = open(local_path, 'rb')
    ftp.storbinary('STOR ' + remote_path, fp, buff_size)
    ftp.set_debuglevel(0)
    fp.close()


def main():
    ftp = ftp_connect('mrirc.psych.ac.cn', 'ftpdownload', 'FTPDownload')
    ftp.cwd('sharing/RfMRIMaps')
    projects = ['ABIDE', 'ABIDE2', 'FCP']

    basic_dir_path = ''
    for project in projects:
        file_list = ftp.cwd('/sharing/RfMRIMaps/{:s}'.format(project))
        dir_path = os.path.join(basic_dir_path, project)
        for file_name in file_list:
            file_path = os.path.join(dir_path, file_name)
            if os.path.exists(file_path):
                print('File {:s} already exist.'.format(file_name))
            else:
                print('Starting download file {:s}...\r\n'.format(file_name))
                continue
            download_file(ftp, file_name, file_path)
    ftp.quit()


if __name__ == '__main__':
    main()


import sklearn.covariance.empirical_covariance_
