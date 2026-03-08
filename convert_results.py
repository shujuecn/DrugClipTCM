from rdkit import Chem
from rdkit.Chem import AllChem

# 1. 设置输入输出文件名
input_file = "results/ranked_compounds.txt"  # 你的结果文件路径
output_file = "results/top_hits_3d.sdf"      # 输出的 3D 结构文件

# 2. 读取前 N 个分子 (比如取前 50 个最好的)
top_n = 10
molecules = []

print(f"正在读取 {input_file} 并转换前 {top_n} 个分子...")

with open(input_file, "r") as f:
    lines = f.readlines()

    # 创建 SDWriter 用于写入 SDF 文件
    writer = Chem.SDWriter(output_file)

    count = 0
    for line in lines:
        if count >= top_n:
            break

        parts = line.strip().split()
        if len(parts) < 2:
            continue

        smiles = parts[0]
        score = parts[1]

        # A. 从 SMILES 创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # B. 给分子起个名字 (比如 Rank_1_Score_0.84)
        mol.SetProp("_Name", f"Rank_{count+1}_Score_{score}")
        mol.SetProp("DrugCLIP_Score", score)

        # C. 【关键】加氢并生成 3D 构象
        mol = Chem.AddHs(mol) # 加氢原子
        try:
            # 尝试生成 3D 坐标
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol) # 简单的力场优化

            # D. 写入文件
            writer.write(mol)
            count += 1
        except Exception:
            print(f"分子 {smiles} 生成 3D 结构失败，跳过。")
            continue

    writer.close()

print(f"转换完成！已生成 {count} 个分子的 3D 结构。")
print(f"请使用 PyMOL 打开: {output_file}")
