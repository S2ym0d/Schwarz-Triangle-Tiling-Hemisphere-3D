# Schwarz Triangle Hemisphere 3D Model Generator

This project generates a 3D-printable, perforated model based on Schwarz triangle tiling of the upper hemisphere model of the hyperbolic space.

🔹 **Inspired by the mathematical art and visualizations of [Prof. Henry Segerman](https://www.thingiverse.com/thing:1608550).**

## 🔧 Project Structure

- `hemisphere_schwarz_modelgen.py`  
  Core module with functions to generate the 3D model.

- `Schwarz_Triangle_Hemisphere_Tiling_3D_Model_Generator.ipynb`  
  A step-by-step Jupyter notebook that guides the user through parameter selection and model generation.

- `requirements.txt`  
  Python dependencies (`numpy`, `trimesh`).

## 📦 Installation

Install required Python packages:

numpy
trimesh

pip install -r requirements.txt

## 🚀 Usage

1. Open the notebook.

2. Set the parameter values to define the desired tiling.

3. Run the model generation cell.

4. The resulting tiling model is saved as an .stl file

## 🧠 Notes

The generated tiling model may require additional post-processing depending on Your use case. For example:

- Adding supports to improve printability.
- Creating a hole at the edge of the hemisphere to insert a light source — useful if You're planning to project shadows in a Poincaré half-plane model.