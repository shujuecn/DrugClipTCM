cd results

# 1. 提取第一个模型 (Split)
obabel -ipdbqt top1_hit_docked.pdbqt -opdb -O best_pose.pdb -m

# 上面命令会生成 best_pose1.pdb, best_pose2.pdb ... 我们只要 best_pose1.pdb
mv best_pose1.pdb ligand.pdb
rm best_pose*.pdb

# 2. 关键：加氢 (Vina 可能会省略非极性氢，MD 必须全原子)
# -p 7.4 模拟生理环境下的质子化状态
obabel -ipdb ligand.pdb -omol2 -O ligand.mol2 -h -p 7.4


# -i: 输入分子
# -c: 电荷计算方法 (gas=Gasteiger, bcc=AM1-BCC推荐)
# -n: 分子净电荷 (设为 0，如果是带电分子需手动指定，如 -1)
# -a: 原子类型 (gaff)
# 运行时间：1m 25s
acpype -i ligand.mol2 -c bcc -n 0 -a gaff

# --add-atoms=heavy: 专门补全缺失的重原子
# --add-residues: 补全缺失的残基（如果有）
pdbfixer protein_raw.pdb --output=protein_fixed.pdb --add-atoms=heavy --add-residues --ph=7.0

# 1. 清洗蛋白：删除氢原子 (-d)，只保留蛋白部分
# -xr 忽略受体中的非标准残基（防止干扰）
obabel protein_fixed.pdb -O protein_clean.pdb -d -xr

# -ff 指定力场 (amber99sb-ildn)
# -water 指定水模型 (tip3p)
gmx pdb2gmx -f protein_clean.pdb -o protein.gro -water tip3p -ff amber99sb-ildn

# 省略中间大量手动替换操作
perl prepare_complex.pl

# 1. 定义盒子
gmx editconf -f complex.gro -o newbox.gro -c -d 1.0 -bt cubic

# 2. 加水
gmx solvate -cp newbox.gro -cs spc216.gro -o solvated.gro -p topol.top

# 因为你的系统现在全是水和蛋白，可能带电（Total charge 不为 0），GROMACS 跑模拟要求系统电中性。
wget http://www.mdtutorials.com/gmx/lysozyme/Files/ions.mdp

# 这一步是关键检验！
# 如果 topol.top 里没写 LIG，或者原子数对不上，这一步会立马报错
gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 1

# 1. 替换溶剂为离子
# 输入 "SOL" (通常是第 13 组)，把水分子替换成钠/氯离子
# -neutral 会自动计算需要多少正负离子来平衡系统
echo "SOL" | gmx genion -s ions.tpr -o solvated_ions.gro -p topol.top -pname NA -nname CL -neutral

# 刚组装好的系统，原子之间可能靠得太近（尤其是小分子和蛋白的边缘），会有巨大的排斥力。我们需要让它们“放松”一下。
wget -nc http://www.mdtutorials.com/gmx/lysozyme/Files/minim.mdp
gmx grompp -f minim.mdp -c solvated_ions.gro -p topol.top -o em.tpr -maxwarn 1

# 这次运行会非常快（可能几秒到几分钟）。你会看到 Potential Energy（势能）迅速下降，最后变为一个很大的负数。那就说明系统已经“舒服”了。
gmx mdrun -v -deffnm em


# 下载 nvt.mdp
wget -nc http://www.mdtutorials.com/gmx/lysozyme/Files/nvt.mdp

# 【重要微调】
# 现在的 nvt.mdp 默认设置 temperature coupling 分组是 "Protein Non-Protein"
# 你的系统里有 Protein, Ligand, Water, Ions。
# GROMACS 会自动把 Ligand+Water+Ions 归类为 "Non-Protein"，这通常是没问题的。
# 这里我们稍微改一下步数，让它跑快点（默认是 50000 步，即 100ps）
# 如果你想测试一下，可以不动它。

# -f: 参数文件
# -c: 输入结构 (EM 跑完的结果)
# -r: 位置限制参考结构 (还是 EM 的结果)
# -p: 拓扑文件
# -o: 输出 tpr
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr


# -v: 显示进度
# -deffnm nvt: 输出文件都叫 nvt.*
# -nb gpu: 使用 GPU 加速计算
gmx mdrun -v -deffnm nvt -nb gpu

# 第四步：NPT 平衡 (控制压强)
wget -nc http://www.mdtutorials.com/gmx/lysozyme/Files/npt.mdp
# 注意输入文件都变了
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -v -deffnm npt -nb gpu

# 关键检查 (验证密度)
# 运行以下命令，并在提示时输入数字 Density 对应的编号（通常是 22 左右，看屏幕提示）：
# 对于常温常压下的水体系，密度应该在 1000 kg/m³ 附近（比如 990 ~ 1020 都是正常的）。
gmx energy -f npt.edr -o density.xvg

# 成品模拟 (Production MD)
wget -nc http://www.mdtutorials.com/gmx/lysozyme/Files/md.mdp

# 1. 设置步数：50ns = 25,000,000 步 (步长 2fs)
sed -i 's/^nsteps.*/nsteps = 25000000/' md.mdp

# 2. 设置保存频率：100ps = 50,000 步
sed -i 's/^nstxout-compressed.*/nstxout-compressed = 50000/' md.mdp  # 压缩轨迹 (xtc)
sed -i 's/^nstenergy.*/nstenergy = 50000/' md.mdp                  # 能量数据 (edr)
sed -i 's/^nstlog.*/nstlog = 50000/' md.mdp                        # 日志 (log)


# 注意：成品模拟通常不再需要位置限制（也就是不加 -r 参数了），让蛋白和小分子自由动起来！
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr

# 这里的 -pme gpu 和 -update gpu 是让 4090 接管所有计算任务
gmx mdrun -v -deffnm md_0_1 -nb gpu -pme gpu -update gpu


# 1. 消除跳跃 (No-jump) 并让蛋白居中
# 选择组的时候：
# Select group for centering: 选 1 (Protein)
# Select group for output:    选 0 (System)
gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_center.xtc -pbc mol -center

# 打开 PyMOL。
# 加载结构文件： File -> Open -> 选择 npt.gro (或者 md_0_1.gro，如果有的话)。
# 加载轨迹文件： File -> Open -> 选择刚才生成的 md_center.xtc。
# 注意：千万别加载原始的 md_0_1.xtc，要看修正后的。
# 播放动画： 点击右下角的播放按钮。


# 计算蛋白骨架的 RMSD
# Group Selection: 选 4 (Backbone) 对 4 (Backbone)
gmx rms -s md_0_1.tpr -f md_center.xtc -o rmsd_protein.xvg -tu ns

# 计算小分子的 RMSD (相对于蛋白)
# Group Selection: 选 "ligand" (或者 LIG 对应的数字) 对 "ligand"
# 注意：这一步可能需要特殊的 index 文件，或者直接选 ligand
# 输入 13 (即 UNL)
gmx rms -s md_0_1.tpr -f md_center.xtc -o rmsd_ligand.xvg -tu ns

# -on: 输出接触数量随时间的变化 (number of contacts)
# -d 0.35: 距离截断值设为 0.35 nm (标准结合距离)
# -group: 开启分组模式
gmx mindist -s md_0_1.tpr -f md_center.xtc -on contacts.xvg -d 0.35 -group

# -res: 计算每个残基的平均值
# 选择 3 (C-alpha) 进行计算
gmx rmsf -s md_0_1.tpr -f md_center.xtc -o rmsf.xvg -res

# 绘制三张图
python plot_analysis.py

# 选 System 会把蛋白、配体、离子、水分子全导出来。虽然文件大一点，但最保险，绝对不会丢东西。
# 导出第 0 帧 (初始状态)
gmx trjconv -s md_0_1.tpr -f md_center.xtc -o start.pdb -dump 0
# 导出第 50 ns (最终稳定状态)
gmx trjconv -s md_0_1.tpr -f md_center.xtc -o final.pdb -dump 50000

# 计算回转半径
# Select group: 1 (Protein)
gmx gyrate -s md_0_1.tpr -f md_center.xtc -o gyrate_ns.xvg -tu ns

# python3 plot_fel_advance.py --rmsd rmsd_ligand.xvg --rg gyrate_ns.xvg --temp 310 --bins 100 --tmin 8.5 --out_prefix FEL_rg_rmsd
# python3 plot_fel_2var.py \
#   --x rmsd_ligand.xvg \
#   --y gyrate.xvg \
#   --temp 310 --bins 100 \
#   --tmin 8.5 \
#   --xlabel "RMSD (nm)" \
#   --ylabel "Rg (nm)" \
#   --out_prefix FEL_rg_rmsd


# 做 FEL 前的关键一步：用“对齐后的轨迹”
# 你现在有 md_center.xtc，但 FEL（尤其 RMSD）建议用“对蛋白骨架拟合”的轨迹，避免把整体平移/旋转引入分布。
# 示例（选择时一般：center 选 Protein，output 选 System；fit 选 Backbone）：
# 1) 先居中去 PBC
gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_nopbc.xtc -pbc mol -center
# 2) 再对蛋白骨架拟合（得到用于分析的轨迹）
gmx trjconv -s md_0_1.tpr -f md_nopbc.xtc -o md_fit.xtc -fit rot+trans

# 通常用 C-alpha 做 PCA（论文最常见，降噪更稳）
# “Select group for covariance analysis”：选 C-alpha
# “Select group for least squares fit”：选 C-alpha（或 Backbone 也行，但建议一致）
gmx covar -s md_0_1.tpr -f md_fit.xtc -o eigenval.xvg -v eigenvec.trr -av average.pdb

# PC1
gmx anaeig -s md_0_1.tpr -f md_fit.xtc -v eigenvec.trr -first 1 -last 1 -proj pc1.xvg -tu ns
# PC2
gmx anaeig -s md_0_1.tpr -f md_fit.xtc -v eigenvec.trr -first 2 -last 2 -proj pc2.xvg -tu ns
# merge
# rmsd_ligand.xvg + gyrate_ns.xvg -> merge_rmsd_gyrate.xyg
perl merge_rmsd_gyrate.pl

# 2D自由能形貌图
gmx sham -f merge_rmsd_gyrate.xvg -ls gibbs.xpm -nlevels 50
python xpm2png.py -f gibbs.xpm -show no -ip yes -o gibbs.png
