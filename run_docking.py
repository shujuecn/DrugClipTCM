import os
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation

# 假设你已经安装了 vina (sudo apt install vina 或 conda install -c conda-forge vina)

def dock_one_molecule(pocket_pdb, smiles, output_name):
    print(f"正在对接分子: {output_name} ...")

    # 1. 准备配体 (Ligand) -> PDBQT
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    # 使用 Meeko 准备 PDBQT (这是 Vina 认识的格式)
    preparator = MoleculePreparation()
    preparator.prepare(mol)
    ligand_pdbqt_string = preparator.write_pdbqt_string()

    with open("./results/top1_ligand.pdbqt", "w") as f:
        f.write(ligand_pdbqt_string)

    # 2. 准备受体 (Receptor) -> PDBQT
    # 这里偷个懒，通常需要用 MGLTools 或 openbabel 转，假设你已经有了或者用 openbabel
    # 简单命令: obabel pocket.pdb -xr -O receptor.pdbqt
    os.system(f"obabel {pocket_pdb} -xr -O ./results/top1_receptor.pdbqt")

    # 3. 计算口袋中心 (Box Center)
    # 简单的取几何中心
    pmol = Chem.MolFromPDBFile(pocket_pdb)
    conf = pmol.GetConformer()
    pts = conf.GetPositions()
    center = pts.mean(axis=0)

    print(f"口袋中心: {center}")

    # 4. 运行 Vina
    # --center_x/y/z: 口袋位置
    # --size_x/y/z: 搜索盒子大小 (20埃足够了)
    cmd = (
        f"vina --receptor ./results/top1_receptor.pdbqt --ligand ./results/top1_ligand.pdbqt "
        f"--center_x {center[0]:.3f} --center_y {center[1]:.3f} --center_z {center[2]:.3f} "
        f"--size_x 20 --size_y 20 --size_z 20 "
        f"--out {output_name}_docked.pdbqt --cpu 8"
    )

    os.system(cmd)
    print(f"对接完成！结果在 {output_name}_docked.pdbqt")

# === 用法示例 ===
# 替换成你刚转换出来的口袋 PDB 路径
POCKET_PDB = "/mnt/e/Workspace/260217-DrugClip/files/retrieval/converted_pockets/7ksi.pdb"
# 替换成 ranked_compounds.txt 里的第一名 SMILES
TOP1_SMILES = "CCOC(=O)c1ccc(NC(=S)Nc2ccn(Cc3c(F)c(F)c(F)c(F)c3F)n2)cc1"

dock_one_molecule(POCKET_PDB, TOP1_SMILES, "./results/top1_hit")
