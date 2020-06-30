# Unsmart-Traffic-Signal-using-DRL
这个环境好像只能在linux下才能跑,不过termios后面也没用到,如果window下装好环境理论上也是没问题的.主要是把环境装好、环境变量SUMO_HOME配置好.



SUMO安装的参考链接:
https://blog.csdn.net/zhixiting5325/article/details/79302244
https://sumo.dlr.de/docs/Installing/Linux_Build.html#Building_the_SUMO_binaries_with_cmake
https://blog.csdn.net/sinat_28199083/article/details/87935933?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase

coppeliasim 4_0_0-2下载地址
https://aur.archlinux.org/packages/coppeliasim/


环境变量参考配置()
export COPPELIASIM_ROOT=/home/j/program/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export SUMO_HOME="/home/j/program/sumo"     #5星配置这个就可以了

export VREP_ROOT="$/home/j/program/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04"

安装过程中出现了很多问题,编译安装还要依赖系统中的一些库文件,5星的环境感觉跟4星的环境一样,环境像套娃一样,一层包一层的,时间都是花在配环境上了

我是从1星做到5星的,在训练1-3星时就花了好几天时间,效果也都没想象的那么好,4星\5星的环境也配了好几天,眼看马上截止了,就只能完成个最简单草草了事了



DQN

make_env.py===========>>>>>>>>>>创建环境

DQNagent.py=====>>>>>>agent

DQNmodel.py=====>>>>>>model

replay_memory.py======>>>>>训练经验池

DQNtrain.py=============>>>>>>>>>>>开始训练

DQNtest.py=============>>>>>>>>>>>>测试打印

DQN训练结果


DDPG方法没搞好 = =#

开始的完全不收敛,一直随机,后来改完训练一段时间绿灯保持在没车的路口了....

make_env.py===========>>>>>>>>>>创建环境

paddle_DDPGagent.py=====>>>>>>agent

paddle_DDPGmodel.py=====>>>>>>model

replay_memory.py======>>>>>训练经验池

train.py=============>>>>>>>>>>>开始训练

test.py=============>>>>>>>>>>>>测试打印




