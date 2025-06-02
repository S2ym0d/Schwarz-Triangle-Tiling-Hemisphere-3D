
import numpy as np
import trimesh
from collections import defaultdict, deque

def triangulated_hemisphere(subdiv:  int) ->  tuple[np.ndarray,  np.ndarray]:
  sphere: trimesh.Trimesh = trimesh.creation.icosphere(subdivisions = subdiv, radius=1.0)

  vertices: np.ndarray = np.array(sphere.vertices)
  faces: np.ndarray = np.array(sphere.faces)

  z_mask: np.ndarray = vertices[:, 2] >= 0

  face_mask: np.ndarray = z_mask[faces].all(axis = 1)

  hemisphere_faces: np.ndarray = faces[face_mask]

  unique_indices, inverse = np.unique(hemisphere_faces.flatten(), return_inverse=True)
  hemisphere_vertices = vertices[unique_indices]
  hemisphere_faces = inverse.reshape(hemisphere_faces.shape)

  return hemisphere_vertices, hemisphere_faces

def is_hyperbolic_schwarz_triangle(p: int, q: int, r: int) -> bool:
  if p < 2 or q < 2 or r < 2:
    return False

  if p * q + p * r + q * r < p * q * r:
    return True
  return False

def validate_user_input(subdiv:int, pqr_values: tuple[int, int, int], vertices_disks_radiuses: np.ndarray, sphere_radius: float, sphere_thickness: float, border_height: float):
  if not is_hyperbolic_schwarz_triangle(pqr_values[0], pqr_values[1], pqr_values[2]):
    print('Schwarz triangle is not hyperbolic')
  
  if not isinstance(subdiv, int):
    print('Subdivision must be an integer')

  if vertices_disks_radiuses[0] < 0 or vertices_disks_radiuses[1] < 0 or vertices_disks_radiuses[2] < 0:
    print('Vertices disks radiuses must be non-negative')
  
  if sphere_radius < 0:
    print('Sphere radius must be non-negative')
  
  if sphere_thickness < 0:
    print('Sphere thickness must be non-negative')

  if border_height < 0:
    print('Border height must be non-negative') 

def hyperbolic_schwarz_triangle_vertices(p: int, q: int, r: int) -> np.ndarray:
  p_angle: float = np.pi / p
  q_angle: float = np.pi / q
  r_angle: float = np.pi / r

  a_side_poincare_disk_length: float = np.sqrt((np.cos(q_angle) + np.cos(p_angle + r_angle)) / (np.cos(q_angle) + np.cos(p_angle - r_angle)))
  b_side_poincare_disk_length: float = np.sqrt((np.cos(r_angle) + np.cos(p_angle + q_angle)) / (np.cos(r_angle) + np.cos(p_angle - q_angle)))

  p_vertex: list[float] = [0, 0]
  r_vertex: list[float] = [0, -a_side_poincare_disk_length]
  q_vertex: list[float] = [b_side_poincare_disk_length * np.sin(p_angle), - b_side_poincare_disk_length * np.cos(p_angle)]

  return np.array([p_vertex, q_vertex, r_vertex])

def hyperboloid_model_normal_to_line(P_vertex_poincare_disk: np.ndarray, Q_vertex_poincare_disk: np.ndarray) -> np.ndarray:
 line_constant_coefficient: float = np.cross(P_vertex_poincare_disk, Q_vertex_poincare_disk)
 line_cross_linear_coefficient: np.ndarray = P_vertex_poincare_disk * (np.dot(Q_vertex_poincare_disk, Q_vertex_poincare_disk) + 1) - Q_vertex_poincare_disk * (np.dot(P_vertex_poincare_disk, P_vertex_poincare_disk) + 1)

 return np.array([2 * line_constant_coefficient, line_cross_linear_coefficient[1], -line_cross_linear_coefficient[0]])

Lorentzian_Metric_Tensor: np.ndarray = np.eye(3)
Lorentzian_Metric_Tensor[0, 0] = -1

def hyperboloid_model_line_reflection_matrix(P_vertex_poincare_disk: np.ndarray, Q_vertex_poincare_disk: np.ndarray) -> np.ndarray:
  lorentzian_cross_product: np.ndarray = hyperboloid_model_normal_to_line(P_vertex_poincare_disk, Q_vertex_poincare_disk)
  lorentzian_cross_product[0] = -lorentzian_cross_product[0]

  lorentzian_cross_product_norm = np.sqrt(np.dot(lorentzian_cross_product, np.matmul(Lorentzian_Metric_Tensor, lorentzian_cross_product)))

  lorentzian_cross_product_normalized = lorentzian_cross_product / lorentzian_cross_product_norm

  reflection_matrix = np.eye(3) - 2 * np.matmul(np.tensordot(lorentzian_cross_product_normalized, lorentzian_cross_product_normalized, axes = 0), Lorentzian_Metric_Tensor)

  return reflection_matrix

def find_fundammental_coordinates(point_hyperboloid_model: np.ndarray, sides_normals: tuple[np.ndarray, np.ndarray, np.ndarray], sides_reflection_matrices: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, bool]:
  MAX_ITER: int = 100
  eps: float = 0.000_001

  oddeven: bool = True

  fundamental_coordinates: np.ndarray = point_hyperboloid_model
  for i in range(MAX_ITER):
    if np.dot(fundamental_coordinates, sides_normals[0]) < -eps:
      fundamental_coordinates = np.matmul(sides_reflection_matrices[0], fundamental_coordinates)
    elif np.dot(fundamental_coordinates, sides_normals[1]) < -eps:
      fundamental_coordinates = np.matmul(sides_reflection_matrices[1], fundamental_coordinates)
    elif np.dot(fundamental_coordinates, sides_normals[2]) < -eps:
      fundamental_coordinates = np.matmul(sides_reflection_matrices[2], fundamental_coordinates)
    else:
      return (fundamental_coordinates, oddeven)
    oddeven = not oddeven
  return (fundamental_coordinates, oddeven)

def hyperboloid_model_to_hemisphere_model(point_hyperboloid_model: np.ndarray) -> np.ndarray:
  w = point_hyperboloid_model[0]
  return np.array([point_hyperboloid_model[1] / w, point_hyperboloid_model[2] / w , 1 / w])

def hemisphere_model_to_hyperboloid_model(point_hemisphere_model: np.ndarray) -> np.ndarray:
  z = point_hemisphere_model[2]
  return np.array([1 / z, point_hemisphere_model[0] / z, point_hemisphere_model[1] / z])

def hyperboloid_model_to_poincare_disk(point_hyperboloid_model: np.ndarray) -> np.ndarray:
  return point_hyperboloid_model[1:3] / (1 + point_hyperboloid_model[0])

def is_in_disk(point_PD: np.ndarray, center_PD: np.ndarray, radius: float) -> bool:
  dist_squared = np.dot(point_PD - center_PD, point_PD - center_PD)
  point_squared = np.dot(point_PD, point_PD)
  center_squared = np.dot(center_PD, center_PD)
  return (2 * dist_squared) < (1 - point_squared) * (1 - center_squared) * (np.cosh(radius) - 1)

def is_point_material(point_hemisphere_model: np.ndarray, triangle_vertices: np.ndarray, vertices_disks_radiuses: np.ndarray, border_height: float, sides_normals: tuple[np.ndarray, np.ndarray, np.ndarray], sides_reflection_matrices: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
  if point_hemisphere_model[2] <= border_height:
    return True

  point_hyperboloid_model: np.ndarray = hemisphere_model_to_hyperboloid_model(point_hemisphere_model)

  fundamental_coordinates, oddeven = find_fundammental_coordinates(point_hyperboloid_model, sides_normals, sides_reflection_matrices)

  fundamental_coordinates_PD: np.ndarray = hyperboloid_model_to_poincare_disk(fundamental_coordinates)

  if is_in_disk(fundamental_coordinates_PD, triangle_vertices[0], vertices_disks_radiuses[0]) or is_in_disk(fundamental_coordinates_PD, triangle_vertices[1], vertices_disks_radiuses[1]) or is_in_disk(fundamental_coordinates_PD, triangle_vertices[2], vertices_disks_radiuses[2]):
    return True

  return oddeven

def distinguish_solid_points(points: np.ndarray, triangle_vertices: np.ndarray, vertices_disks_radiuses: np.ndarray, border_height: float) -> np.ndarray:
  normals = (hyperboloid_model_normal_to_line(triangle_vertices[0], triangle_vertices[2]), hyperboloid_model_normal_to_line(triangle_vertices[2], triangle_vertices[1]), hyperboloid_model_normal_to_line(triangle_vertices[1], triangle_vertices[0]))
  reflection_matrices = (hyperboloid_model_line_reflection_matrix(triangle_vertices[0], triangle_vertices[2]), hyperboloid_model_line_reflection_matrix(triangle_vertices[2], triangle_vertices[1]), hyperboloid_model_line_reflection_matrix(triangle_vertices[1], triangle_vertices[0]))

  args = [(p, triangle_vertices, vertices_disks_radiuses, border_height, normals, reflection_matrices) for p in points]

  results: list[bool] = []

  for p in points:
    result = is_point_material(p, triangle_vertices, vertices_disks_radiuses, border_height, normals, reflection_matrices)
    results.append(result)

  return np.array(results)

def make_3d_model(points: np.ndarray, triangles: np.ndarray, is_solid: np.ndarray, sphere_inner_radius: float, sphere_thickness: float) -> trimesh.Trimesh:

  selected_triangles: list[np.ndarray] = []
  for tri_indices in triangles:
    if is_solid[tri_indices].sum() > 2:
      selected_triangles.append(tri_indices)

  edge_count = defaultdict(int)

  for tri in selected_triangles:
        for i in range(3):
            a, b = sorted((tri[i], tri[(i+1)%3]))
            edge_count[(a,b)] += 1

  naked = [e for e,c in edge_count.items() if c==1]
  naked_set = set(map(tuple, naked))

  top_vertices: np.ndarray = points * (sphere_inner_radius + sphere_thickness)
  bottom_vertices: np.ndarray = points * sphere_inner_radius

  all_vertices: np.ndarray = np.vstack((top_vertices, bottom_vertices))

  num_points = points.shape[0]
  bottom_indices_offset = num_points

  mesh_faces: list[list[int]] = []

  for tri_indices in selected_triangles:
    mesh_faces.append(tri_indices.tolist())
    mesh_faces.append([tri_indices[2] + bottom_indices_offset, tri_indices[1] + bottom_indices_offset, tri_indices[0] + bottom_indices_offset])

    for i in range(3):
      v1_top_idx = tri_indices[i]
      v2_top_idx = tri_indices[(i+1)%3]

      a, b = sorted((v1_top_idx, v2_top_idx))

      if (a, b) in naked_set:
       v1_bot_idx = tri_indices[i] + bottom_indices_offset
       v2_bot_idx = tri_indices[(i+1)%3] + bottom_indices_offset

       mesh_faces.append([v1_top_idx, v2_top_idx, v1_bot_idx])
       mesh_faces.append([v2_top_idx, v2_bot_idx, v1_bot_idx])

  mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=all_vertices, faces=np.array(mesh_faces))
  mesh.remove_unreferenced_vertices()
  mesh.remove_duplicate_faces()
  mesh.merge_vertices()

  return mesh

def generate_hemisphere_tiling_model(subdiv:int, pqr_values: tuple[int, int, int], vertices_disks_radiuses: np.ndarray, sphere_radius: float, sphere_thickness: float, border_height: float):
  EPS: float = 0.000_001

  if border_height < EPS:
    border_height = EPS

  border_height = border_height / sphere_radius

  points, triangles = triangulated_hemisphere(subdiv)

  vertices: np.ndarray = hyperbolic_schwarz_triangle_vertices(pqr_values[0], pqr_values[1], pqr_values[2])

  is_solid: np.ndarray = distinguish_solid_points(points, vertices, vertices_disks_radiuses, border_height)

  sphere_inner_radius = sphere_radius - sphere_thickness

  if sphere_inner_radius < EPS:
    sphere_inner_radius = EPS
    sphere_thickness = sphere_radius - EPS

  mesh = make_3d_model(points, triangles, is_solid, sphere_inner_radius, sphere_thickness)
  if len(mesh.faces) > 0:
    name: str = 'model' + '_' + str(pqr_values[0]) + '_' + str(pqr_values[1]) + '_' + str(pqr_values[2]) + '.stl'
    mesh.export(name)
  else:
    print(f"No solid material found for pqr={pqr_values}. No model exported.")
