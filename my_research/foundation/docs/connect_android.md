adb -a nodaemon server start
ssh -p 60022 -R 5037:127.0.0.1:5037 root@163.152.162.139
adb kill-server
adb devices