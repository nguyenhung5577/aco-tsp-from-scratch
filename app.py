import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QTabWidget, QGridLayout, QMessageBox, QGroupBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from artificial_bee_colony import ArtificialBeeColony, GeneticAlgorithm , HillClimbing, SimulatedAnnealing
from fitness_function import (
    ackley_function, rastrigin_function, rosenbrock_function,
    sphere_function, get_function_bounds, get_global_min_pos
)

# --- Matplotlib Canvas Class ---
class MplCanvas(FigureCanvas):
    """
    Lớp dùng để nhúng Matplotlib Figure vào widget PyQt
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout() 

# --- Main Application Window ---
class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trực Quan Hóa Thuật Toán Tối Ưu")
        self.setGeometry(0, 0, 1200, 800)
        
        # Khởi tạo các biến lưu trữ dữ liệu
        self.history_abc = None
        self.best_abc = None
        self.fitness_abc = None
        self.history_ga = None
        self.best_ga = None
        self.fitness_ga = None
        self.history_hc = None 
        self.best_hc = None    
        self.fitness_hc = None 
        self.history_sa = None 
        self.best_sa = None    
        self.fitness_sa = None 
        
        self.current_animation = None 
        
        # Khởi tạo UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Thiết lập giao diện người dùng"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- Cột trái: Tham số và Kết quả ---
        control_panel = QVBoxLayout()
        control_panel.setAlignment(Qt.AlignTop)
        
        # 1. Nhóm chọn Hàm Fitness
        func_group = QGroupBox("1. Chọn Hàm Fitness")
        func_layout = QGridLayout()
        
        self.function_map = {
            "Ackley Function": ackley_function,
            "Rastrigin Function": rastrigin_function,
            "Rosenbrock Function": rosenbrock_function,
            "Sphere Function": sphere_function,
        }
        
        self.func_combo = QComboBox()
        self.func_combo.addItems(self.function_map.keys())
        self.func_combo.setCurrentText("Ackley Function")
        self.func_combo.currentIndexChanged.connect(self._update_bounds_display)

        self.bounds_label = QLabel("Giới hạn: [-32.768, 32.768]") 
        
        func_layout.addWidget(QLabel("Hàm Fitness:"), 0, 0)
        func_layout.addWidget(self.func_combo, 0, 1)
        func_layout.addWidget(self.bounds_label, 1, 0, 1, 2)
        func_group.setLayout(func_layout)
        control_panel.addWidget(func_group)
        
        # 2. Nhóm Tham số Thuật toán
        param_group = QGroupBox("2. Tham số Chung")
        param_layout = QGridLayout()

        self.max_iter_input = QLineEdit("50")
        self.pop_size_input = QLineEdit("40")
        
        param_layout.addWidget(QLabel("Số vòng lặp (Max Iter):"), 0, 0)
        param_layout.addWidget(self.max_iter_input, 0, 1)
        param_layout.addWidget(QLabel("Kích thước quần thể (Pop Size):"), 1, 0)
        param_layout.addWidget(self.pop_size_input, 1, 1)
        
        param_group.setLayout(param_layout)
        control_panel.addWidget(param_group)
        
        # 3. Nút Chạy
        self.run_button = QPushButton("Chạy Tối Ưu Hóa")
        self.run_button.clicked.connect(self._run_optimization)
        control_panel.addWidget(self.run_button)
        
        # 4. Nhóm Kết quả
        result_group = QGroupBox("3. Kết Quả Tốt Nhất")
        result_layout = QGridLayout()

        self.abc_fitness_label = QLabel("ABC Best Fitness: N/A")
        self.abc_solution_label = QLabel("ABC Best Solution: N/A")
        self.ga_fitness_label = QLabel("GA Best Fitness: N/A")
        self.ga_solution_label = QLabel("GA Best Solution: N/A")
        self.hc_fitness_label = QLabel("HC Best Fitness: N/A")
        self.hc_solution_label = QLabel("HC Best Solution: N/A")
        self.sa_fitness_label = QLabel("SA Best Fitness: N/A")
        self.sa_solution_label = QLabel("SA Best Solution: N/A")
        
        result_layout.addWidget(self.abc_fitness_label, 0, 0, 1, 2)
        result_layout.addWidget(self.abc_solution_label, 1, 0, 1, 2)
        result_layout.addWidget(self.ga_fitness_label, 2, 0, 1, 2)
        result_layout.addWidget(self.ga_solution_label, 3, 0, 1, 2)
        result_layout.addWidget(self.hc_fitness_label, 4, 0, 1, 2)
        result_layout.addWidget(self.hc_solution_label, 5, 0, 1, 2)
        result_layout.addWidget(self.sa_fitness_label, 6, 0, 1, 2)
        result_layout.addWidget(self.sa_solution_label, 7, 0, 1, 2)
        
        result_group.setLayout(result_layout)
        control_panel.addWidget(result_group)
        
        control_panel.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        main_layout.addLayout(control_panel, 1) 
        
        # --- Cột phải: Trực quan hóa ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, 4) 
        
        # Khởi tạo các tab
        self.tab_3d = QWidget()
        self.tab_convergence = QWidget()
        self.tab_contour = QWidget()
        self.tab_animation_abc = QWidget() 
        self.tab_animation_ga = QWidget()
        self.tab_animation_hc = QWidget()
        self.tab_animation_sa = QWidget()
        
        self.tab_widget.addTab(self.tab_3d, "Đồ thị 3D")
        self.tab_widget.addTab(self.tab_convergence, "Đồ thị Hội tụ (Best Fitness)")
        self.tab_widget.addTab(self.tab_contour, "Biểu đồ Đường Đồng Mức 2D (Kết quả cuối)")
        self.tab_widget.addTab(self.tab_animation_abc, "Animation 2D (ABC)") 
        self.tab_widget.addTab(self.tab_animation_ga, "Animation 2D (GA)")
        self.tab_widget.addTab(self.tab_animation_hc, "Animation 2D (HC)")
        self.tab_widget.addTab(self.tab_animation_sa, "Animation 2D (SA)")

        # Khởi tạo Matplotlib Canvases
        self.canvas_3d = MplCanvas(self.tab_3d, width=5, height=5)
        self.canvas_convergence = MplCanvas(self.tab_convergence, width=5, height=5)
        self.canvas_contour = MplCanvas(self.tab_contour, width=5, height=5)
        self.canvas_animation_abc = MplCanvas(self.tab_animation_abc, width=5, height=5)
        self.canvas_animation_ga = MplCanvas(self.tab_animation_ga, width=5, height=5)
        self.canvas_animation_hc = MplCanvas(self.tab_animation_hc, width=5, height=5)
        self.canvas_animation_sa = MplCanvas(self.tab_animation_sa, width=5, height=5)

        # Đặt layout cho các tab
        layout_3d = QVBoxLayout(self.tab_3d)
        layout_3d.addWidget(self.canvas_3d)
        layout_convergence = QVBoxLayout(self.tab_convergence)
        layout_convergence.addWidget(self.canvas_convergence)
        layout_contour = QVBoxLayout(self.tab_contour)
        layout_contour.addWidget(self.canvas_contour)
        
        layout_animation_abc = QVBoxLayout(self.tab_animation_abc) 
        layout_animation_abc.addWidget(self.canvas_animation_abc)
        layout_animation_ga = QVBoxLayout(self.tab_animation_ga)
        layout_animation_ga.addWidget(self.canvas_animation_ga)
        layout_animation_hc = QVBoxLayout(self.tab_animation_hc)
        layout_animation_hc.addWidget(self.canvas_animation_hc)
        layout_animation_sa = QVBoxLayout(self.tab_animation_sa)
        layout_animation_sa.addWidget(self.canvas_animation_sa)
        
        # Kết nối sự kiện chuyển tab để quản lý animation
        self.tab_widget.currentChanged.connect(self._stop_all_animations)

        # Cập nhật giới hạn ban đầu
        self._update_bounds_display()
        self._plot_3d() # Vẽ đồ thị 3D ban đầu

    def _stop_all_animations(self):
        """Dừng tất cả các animation đang chạy."""
        # Dừng animation cũ nếu có
        if self.current_animation:
            self.current_animation.event_source.stop()
        self.current_animation = None # Reset
        
        # Chỉ chạy lại animation khi tab animation được chọn
        current_index = self.tab_widget.currentIndex()
        if current_index == 3:
            self._plot_animation_2d(self.function_map[self.func_combo.currentText()], self.current_bounds[0], self.current_bounds[1], self.history_abc, "ABC", self.canvas_animation_abc)
        elif current_index == 4:
            self._plot_animation_2d(self.function_map[self.func_combo.currentText()], self.current_bounds[0], self.current_bounds[1], self.history_ga, "GA", self.canvas_animation_ga)
        elif current_index == 5:
            self._plot_animation_2d(self.function_map[self.func_combo.currentText()], self.current_bounds[0], self.current_bounds[1], self.history_hc, "HC", self.canvas_animation_hc, single_point=True)
        elif current_index == 6:
            self._plot_animation_2d(self.function_map[self.func_combo.currentText()], self.current_bounds[0], self.current_bounds[1], self.history_sa, "SA", self.canvas_animation_sa, single_point=True)


    def _update_bounds_display(self):
        """Cập nhật nhãn giới hạn khi thay đổi hàm fitness"""
        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = get_function_bounds(selected_func)
        self.current_bounds = (l_bound, u_bound)
        self.bounds_label.setText(f"Giới hạn: [{l_bound}, {u_bound}]")
        
        # Tự động vẽ lại 3D khi thay đổi hàm
        self._plot_3d()

    def _get_parameters(self):
        """Lấy các tham số từ input và xử lý lỗi"""
        try:
            max_iters = int(self.max_iter_input.text())
            pop_size = int(self.pop_size_input.text())
            if max_iters <= 0 or pop_size <= 0:
                raise ValueError("Số lượng phải lớn hơn 0.")
            if pop_size % 2 != 0:
                pop_size += 1 # Đảm bảo kích thước quần thể chẵn cho ABC chia đều
            
            return max_iters, pop_size
        except ValueError as e:
            QMessageBox.critical(self, "Lỗi Tham số", f"Tham số không hợp lệ: {e}. Vui lòng kiểm tra lại (Max Iter, Pop Size).")
            return None, None

    def _run_optimization(self):
        """Chạy thuật toán tối ưu hóa ABC, GA, HC, SA"""
        max_iters, pop_size = self._get_parameters()
        if max_iters is None:
            return

        selected_func_name = self.func_combo.currentText()
        selected_func = self.function_map[selected_func_name]
        l_bound, u_bound = self.current_bounds
        
        self.run_button.setEnabled(False)
        self.run_button.setText("Đang chạy thuật toán...")

        self._stop_all_animations() # Dừng animation cũ nếu có

        try:
            # --- 1. ABC Run ---
            abc_solver = ArtificialBeeColony(
                fitness_func=selected_func,
                lower_bound=l_bound,
                upper_bound=u_bound,
                max_iterations=max_iters,
                num_employed=pop_size // 2,
                num_onlooker=pop_size // 2
            )
            self.history_abc, self.best_abc, self.fitness_abc = abc_solver.run()

            # --- 2. GA Run ---
            ga_solver = GeneticAlgorithm(
                fitness_func=selected_func,
                lower_bound=l_bound,
                upper_bound=u_bound,
                max_iterations=max_iters,
                population_size=pop_size,
                crossover_rate=0.8,
                mutation_rate=0.05
            )
            self.history_ga, self.best_ga, self.fitness_ga = ga_solver.run()

            # --- 3. Hill Climbing Run ---
            hc_solver = HillClimbing(
                fitness_func=selected_func,
                lower_bound=l_bound,
                upper_bound=u_bound,
                max_iterations=max_iters,
                step_size=0.01 * (u_bound - l_bound) # Step size nhỏ hơn
            )
            self.history_hc, self.best_hc, self.fitness_hc = hc_solver.run()

            # --- 4. Simulated Annealing Run ---
            sa_solver = SimulatedAnnealing(
                fitness_func=selected_func,
                lower_bound=l_bound,
                upper_bound=u_bound,
                max_iterations=max_iters,
                initial_temp=10.0,
                cooling_rate=0.95
            )
            self.history_sa, self.best_sa, self.fitness_sa = sa_solver.run()


            # Cập nhật kết quả
            self.abc_fitness_label.setText(f"ABC Best Fitness: {self.fitness_abc:.6f}")
            self.abc_solution_label.setText(f"ABC Best Solution (X): [{self.best_abc[0]:.4f}, {self.best_abc[1]:.4f}]")
            self.ga_fitness_label.setText(f"GA Best Fitness: {self.fitness_ga:.6f}")
            self.ga_solution_label.setText(f"GA Best Solution (X): [{self.best_ga[0]:.4f}, {self.best_ga[1]:.4f}]")
            self.hc_fitness_label.setText(f"HC Best Fitness: {self.fitness_hc:.6f}")
            self.hc_solution_label.setText(f"HC Best Solution (X): [{self.best_hc[0]:.4f}, {self.best_hc[1]:.4f}]")
            self.sa_fitness_label.setText(f"SA Best Fitness: {self.fitness_sa:.6f}")
            self.sa_solution_label.setText(f"SA Best Solution (X): [{self.best_sa[0]:.4f}, {self.best_sa[1]:.4f}]")

            # Vẽ lại các đồ thị
            self._plot_convergence(max_iters)
            self._plot_contour(selected_func, l_bound, u_bound)
            
            # Khởi tạo animation mặc định
            self._plot_animation_2d(selected_func, l_bound, u_bound, self.history_abc, "ABC", self.canvas_animation_abc) 
            
            self.tab_widget.setCurrentIndex(3) # Chuyển sang tab Animation ABC
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Tính toán", f"Đã xảy ra lỗi trong quá trình chạy thuật toán: {e}")
            print(f"Error during optimization run: {e}")
        finally:
            self.run_button.setEnabled(True)
            self.run_button.setText("Chạy Tối Ưu Hóa")

    def _plot_3d(self):
        """Vẽ đồ thị bề mặt 3D của hàm fitness"""
        selected_func = self.function_map[self.func_combo.currentText()]
        func_name = self.func_combo.currentText()
        l_bound, u_bound = self.current_bounds
        
        ax = self.canvas_3d.axes
        ax.cla() 
        
        try:
            x = np.linspace(l_bound, u_bound, 500)
            y = np.linspace(l_bound, u_bound, 500)
            x_meshgrid, y_meshgrid = np.meshgrid(x, y)
            
            z = selected_func([x_meshgrid, y_meshgrid]) 

            self.canvas_3d.fig.clear()
            ax = self.canvas_3d.fig.add_subplot(111, projection='3d')
            
            ax.plot_surface(x_meshgrid, y_meshgrid, z,
                            cmap='viridis', edgecolor='none', alpha=0.8)

            global_min_pos = get_global_min_pos(selected_func)
            if global_min_pos:
                global_min_val = selected_func(global_min_pos)
                ax.scatter(global_min_pos[0], global_min_pos[1], global_min_val,
                        color='red', marker='*', s=200, label='Global Minimum', zorder=5)

            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_zlabel(f'Giá trị Fitness $f(x)$')
            ax.set_title(f"Đồ thị 3D Hàm {func_name}")
            ax.legend()
            self.canvas_3d.draw()
            
        except Exception as e:
            ax.set_title("Lỗi vẽ đồ thị 3D")
            print(f"Error plotting 3D surface: {e}")
            self.canvas_3d.draw()

    def _plot_convergence(self, max_iters):
        """Vẽ đồ thị hội tụ (Best Fitness theo Iteration) cho 4 thuật toán"""
        ax = self.canvas_convergence.axes
        ax.cla() 

        if self.history_abc is None: # Chỉ cần kiểm tra 1 thuật toán đã chạy chưa
            ax.text(0.5, 0.5, "Vui lòng chạy tối ưu hóa trước", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Đồ thị Hội tụ")
            self.canvas_convergence.draw()
            return
        
        selected_func_name = self.func_combo.currentText()
        selected_func = self.function_map[selected_func_name]
        
        # Tính toán lịch sử Best Fitness cho 4 thuật toán
        abc_best_fitness_history = [np.min([selected_func(p) for p in positions]) for positions in self.history_abc]
        ga_best_fitness_history = [np.min([selected_func(p) for p in positions]) for positions in self.history_ga]
        # HC và SA luôn có quần thể kích thước 1, nên chỉ cần lấy giá trị fitness đó
        hc_best_fitness_history = [selected_func(positions[0]) for positions in self.history_hc]
        sa_best_fitness_history = [selected_func(positions[0]) for positions in self.history_sa]
        
        iterations = np.arange(len(abc_best_fitness_history)) 
        
        ax.plot(iterations, abc_best_fitness_history, label=f'ABC (Best: {self.fitness_abc:.4f})', color='blue')
        ax.plot(iterations, ga_best_fitness_history, label=f'GA (Best: {self.fitness_ga:.4f})', color='red')
        ax.plot(iterations, hc_best_fitness_history, label=f'HC (Best: {self.fitness_hc:.4f})', color='green')
        ax.plot(iterations, sa_best_fitness_history, label=f'SA (Best: {self.fitness_sa:.4f})', color='yellow')
        
        ax.set_xlabel("Vòng lặp (Iteration)")
        ax.set_ylabel("Giá trị Fitness Tốt Nhất")
        ax.set_title(f"Đồ thị Hội tụ - Hàm {selected_func_name}")
        ax.legend()
        ax.grid(True)
        self.canvas_convergence.draw()

    def _plot_contour(self, selected_func, l_bound, u_bound):
        """Vẽ biểu đồ đường đồng mức 2D và vị trí cuối cùng của 4 thuật toán"""
        ax = self.canvas_contour.axes
        ax.cla() 

        if self.history_abc is None:
            ax.text(0.5, 0.5, "Vui lòng chạy tối ưu hóa trước", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Biểu đồ Đường Đồng Mức 2D")
            self.canvas_contour.draw()
            return

        x = np.linspace(l_bound, u_bound, 100)
        y = np.linspace(l_bound, u_bound, 100)
        x_meshgrid, y_meshgrid = np.meshgrid(x, y)
        z = selected_func([x_meshgrid, y_meshgrid])

        contour = ax.contourf(x_meshgrid, y_meshgrid, z,
                            levels=50, cmap='viridis', alpha=0.8)

        # Vị trí cuối cùng của ABC (Quần thể)
        final_abc_positions = self.history_abc[-1]
        ax.scatter(final_abc_positions[:, 0], final_abc_positions[:, 1], 
                   c='yellow', s=50, label='Vị trí cuối cùng (ABC)', edgecolor='black', marker='o')

        # Vị trí cuối cùng của GA (Quần thể)
        final_ga_positions = self.history_ga[-1]
        ax.scatter(final_ga_positions[:, 0], final_ga_positions[:, 1], 
                   c='cyan', s=50, label='Vị trí cuối cùng (GA)', edgecolor='black', marker='s', alpha=0.6)
        
        # Vị trí cuối cùng của HC (Điểm)
        final_hc_position = self.history_hc[-1][0]
        ax.plot(final_hc_position[0], final_hc_position[1], 
                   'g^', markersize=10, markeredgecolor='black', label='Vị trí cuối cùng (HC)')
        
        # Vị trí cuối cùng của SA (Điểm)
        final_sa_position = self.history_sa[-1][0]
        ax.plot(final_sa_position[0], final_sa_position[1], 
                   'mv', markersize=10, markeredgecolor='black', label='Vị trí cuối cùng (SA)')


        # Global Minimum
        global_min_pos = get_global_min_pos(selected_func)
        if global_min_pos:
            ax.plot(global_min_pos[0], global_min_pos[1], 'w*', markersize=12,
                    markeredgecolor='black', label='Global Minimum')

        ax.set_xlim(l_bound, u_bound)
        ax.set_ylim(l_bound, u_bound)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(f"Kết quả cuối trên Biểu đồ Đường Đồng Mức - Hàm {self.func_combo.currentText()}")
        ax.legend(loc='upper right')
        self.canvas_contour.draw()

    def _plot_animation_2d(self, func, l_bound, u_bound, positions_history, algorithm_name, canvas, single_point=False):
        """Tạo animation"""
        ax = canvas.axes
        ax.cla()

        if not positions_history:
            ax.text(0.5, 0.5, f"Lịch sử vị trí của {algorithm_name} trống.", ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return

        x = np.linspace(l_bound, u_bound, 100)
        y = np.linspace(l_bound, u_bound, 100)
        x_meshgrid, y_meshgrid = np.meshgrid(x, y)
        z = func([x_meshgrid, y_meshgrid])

        contour = ax.contourf(x_meshgrid, y_meshgrid, z, levels=50, cmap='viridis', alpha=0.8)

        global_min_pos = get_global_min_pos(func)
        if global_min_pos:
            ax.plot(global_min_pos[0], global_min_pos[1], 'w*', markersize=12,
                    markeredgecolor='black', label='Global Minimum')

        if single_point:
            scat, = ax.plot([], [], 'o', color='yellow', markersize=8, label=f'Giải pháp hiện tại ({algorithm_name})', markeredgecolor='black')
            best_point = None
        else:
            scat = ax.scatter([], [], c='yellow', s=50, label=f'Quần thể ({algorithm_name})', edgecolor='black')
            best_point, = ax.plot([], [], 'o', color='red', markersize=8, markeredgecolor='black', label='Best Solution Found')
        
        iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='black', verticalalignment='top')


        ax.set_xlim(l_bound, u_bound)
        ax.set_ylim(l_bound, u_bound)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(f"Animation {algorithm_name} - Hàm {self.func_combo.currentText()}")
        ax.legend(loc='upper right')
        
        def update(frame):
            current_positions = positions_history[frame]
            
            if single_point:
                scat.set_data(np.array([current_positions[0][0]]), np.array([current_positions[0][1]]))
                return_artists = [scat, iteration_text]
            else:
                scat.set_offsets(current_positions)
                current_fitness = np.array([func(p) for p in current_positions])
                current_best_pos = current_positions[np.argmin(current_fitness)]
                best_point.set_data(np.array([current_best_pos[0]]), np.array([current_best_pos[1]]))
                return_artists = [scat, best_point, iteration_text]
            
            iteration_text.set_text(f"Iteration: {frame}/{len(positions_history)-1}")
            
            return return_artists

        self.current_animation = FuncAnimation(
            canvas.fig, 
            update, 
            frames=len(positions_history), 
            interval=150, 
            repeat=True, 
            blit=False
        )
        canvas.draw()

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())