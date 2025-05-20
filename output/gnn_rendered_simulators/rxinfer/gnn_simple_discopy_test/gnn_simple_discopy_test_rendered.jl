using RxInfer


@model function Simple DisCoPy Test Model()

end


# --- Inference ---
# Note: Ensure that data variables (e.g., your_data_variables)
# are defined and loaded in the Julia environment before this script section.
# Example:
# using CSV, DataFrames
# my_data_table = CSV.read("path/to/your/data.csv", DataFrame)
# y_observed_data = my_data_table.y_column
# X_matrix_data = Matrix(my_data_table[!, [:x1_column, :x2_column]])

result = infer(
    model = Simple DisCoPy Test Model
    iterations = 50
)
