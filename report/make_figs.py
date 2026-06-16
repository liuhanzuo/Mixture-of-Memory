import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
# 注册 Noto CJK 中文字体
for fp in ["/usr/share/fonts/google-noto-cjk/NotoSansCJKsc-Regular.otf",
           "/usr/share/fonts/google-noto-cjk/NotoSansCJKsc-Light.otf"]:
    try: fm.fontManager.addfont(fp)
    except: pass
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus']=False
OUT="report/figs/"

# 图1: coef 倒U曲线 (babilong qa5 长程, mass+蒸馏叠加)
fig,ax=plt.subplots(figsize=(5.2,3.2))
coef=[0.3,0.5,0.7,2.0]
v16=[11,13,10,6]; v32=[8,9,8,5]
ax.plot(coef,v16,'o-',label='16k',lw=2,ms=7)
ax.plot(coef,v32,'s-',label='32k',lw=2,ms=7)
ax.axhline(8,ls='--',c='gray',lw=1,label='纯蒸馏 32k')
ax.set_xlabel('mass 强度 coef'); ax.set_ylabel('qa5 W0 准确率')
ax.set_title('弱 mass + 蒸馏：长程协同呈倒 U（峰 coef≈0.5）')
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT+'coef_invertedU.pdf'); fig.savefig(OUT+'coef_invertedU.png',dpi=130)

# 图2: 四杠杆 W0 qa5 对照 (长度曲线)
fig,ax=plt.subplots(figsize=(5.6,3.4))
L=[0,1,2,4,8,16,32]; xi=range(len(L))
base=[70,31,53,22,13,8,6]
mass=[78,58,48,28,10,12,7]
dist=[70,59,45,25,15,11,8]
ovl =[70,49,44,25,14,13,9]
ax.plot(xi,base,'--',c='gray',label='baseline',lw=1.5)
ax.plot(xi,mass,'o-',label='mass(coef2)',lw=1.8)
ax.plot(xi,dist,'^-',label='蒸馏',lw=1.8)
ax.plot(xi,ovl,'s-',label='弱mass+蒸馏(0.5)',lw=2.2,c='crimson')
ax.set_xticks(list(xi)); ax.set_xticklabels([f'{x}k' for x in L])
ax.set_xlabel('上下文长度'); ax.set_ylabel('qa5 W0 准确率')
ax.set_title('四类杠杆 W0 readout（BABILong qa5）')
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT+'levers_qa5.pdf'); fig.savefig(OUT+'levers_qa5.png',dpi=130)

# 图3: longbench 能力缺失 (mass coef1 W0 vs base 开卷)
fig,ax=plt.subplots(figsize=(5.8,3.2))
ds=['narrativeqa','qasper','musique','multifieldqa','hotpotqa','2wikimqa']
w0=[2.6,5.2,3.5,12.3,6.5,9.2]; bo=[16.0,13.9,7.0,24.9,9.8,12.2]
x=range(len(ds)); ww=0.38
ax.bar([i-ww/2 for i in x],bo,ww,label='base 开卷',color='steelblue')
ax.bar([i+ww/2 for i in x],w0,ww,label='memory W0(闭卷)',color='salmon')
ax.set_xticks(list(x)); ax.set_xticklabels(ds,rotation=30,ha='right',fontsize=7)
ax.set_ylabel('F1'); ax.set_title('LongBench 真实长文档：能力缺失定位')
ax.legend(fontsize=8); ax.grid(alpha=0.3,axis='y'); fig.tight_layout(); fig.savefig(OUT+'longbench_gap.pdf'); fig.savefig(OUT+'longbench_gap.png',dpi=130)
print("figs done:", __import__('os').listdir(OUT))
