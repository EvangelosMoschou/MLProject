import nbformat as nbf
import os
import shutil

def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def create_notebook_ac():
    # Create cells for Team1-AC.ipynb
    nb_ac = nbf.v4.new_notebook()
    
    # Header cell
    header_text = """# Machine Learning Project - Parts A, B, C
**Team 1**
* Name: [Your Name Here]
* AEM: [Your AEM Here]
"""
    nb_ac.cells.append(nbf.v4.new_markdown_cell(header_text))
    
    # Part A Cell
    part_a_code = read_file('PartA/solution_a.py')
    nb_ac.cells.append(nbf.v4.new_markdown_cell("## Part A: Maximum Likelihood Estimation"))
    nb_ac.cells.append(nbf.v4.new_code_cell(part_a_code))
    
    # Part B Cell
    part_b_code = read_file('PartB/solution_b.py')
    nb_ac.cells.append(nbf.v4.new_markdown_cell("## Part B: Parzen Window Density Estimation"))
    nb_ac.cells.append(nbf.v4.new_code_cell(part_b_code))
    
    # Part C Cell
    part_c_code = read_file('PartC/solution_c.py')
    nb_ac.cells.append(nbf.v4.new_markdown_cell("## Part C: K-Nearest Neighbors Classifier"))
    nb_ac.cells.append(nbf.v4.new_code_cell(part_c_code))
    
    # Write Team1-AC.ipynb
    with open('Team1-AC.ipynb', 'w') as f:
        nbf.write(nb_ac, f)
    print("Created Team1-AC.ipynb")

def create_notebook_d():
    # Create cells for Team1-D.ipynb
    nb_d = nbf.v4.new_notebook()
    
    # Header cell
    header_text = """# Machine Learning Project - Part D
**Team 1**
* Name: [Your Name Here]
* AEM: [Your AEM Here]
"""
    nb_d.cells.append(nbf.v4.new_markdown_cell(header_text))
    
    # Part D Cell
    part_d_code = read_file('PartD/solution_d.py')
    nb_d.cells.append(nbf.v4.new_markdown_cell("## Part D: Classification Challenge"))
    nb_d.cells.append(nbf.v4.new_code_cell(part_d_code))
    
    # Write Team1-D.ipynb
    with open('Team1-D.ipynb', 'w') as f:
        nbf.write(nb_d, f)
    print("Created Team1-D.ipynb")

def main():
    create_notebook_ac()
    create_notebook_d()
    
    # Copy labels1.npy to root if it exists in PartD
    if os.path.exists('PartD/labels1.npy'):
        shutil.copy('PartD/labels1.npy', 'labels1.npy')
        print("Copied labels1.npy to root")

if __name__ == "__main__":
    main()
