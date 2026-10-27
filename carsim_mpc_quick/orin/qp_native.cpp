// Same scaled ADMM and stopping conditions as qp_admm.m; row-major doubles.
// Compile: g++ -std=c++17 -O3 -fPIC -shared qp_native.cpp -o qp_native.so
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" int suspension_qp(
    int n, int m, int max_iter, double rho, double sigma, double abs_tol, double rel_tol,
    const double* a, const double* at, const double* r, const double* f,
    const double* lo, const double* hi, double* x, double* z, double* y, double* stats) {
    if (n <= 0 || m <= 0 || max_iter <= 0) return -1;
    std::vector<double> old_x(n), old_z(m), rhs(n), ax(m), work(m);
    for (int k = 1; k <= max_iter; ++k) {
        std::copy(x, x + n, old_x.begin());
        std::copy(z, z + m, old_z.begin());
        for (int j = 0; j < m; ++j) work[j] = z[j] - y[j];
        for (int i = 0; i < n; ++i) {
            double sum = 0;
            for (int j = 0; j < m; ++j) sum += at[i*m+j] * work[j];
            rhs[i] = -f[i] + sigma*x[i] + rho*sum;
        }
        // Solve R' tmp = rhs, then R x = tmp. R is MATLAB's upper chol(K).
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) rhs[i] -= r[j*n+i]*rhs[j];
            rhs[i] /= r[i*n+i];
        }
        for (int i = n-1; i >= 0; --i) {
            for (int j = i+1; j < n; ++j) rhs[i] -= r[i*n+j]*x[j];
            x[i] = rhs[i]/r[i*n+i];
        }
        double primal = 0, norm_ax = 0, norm_z = 0;
        for (int i = 0; i < m; ++i) {
            double sum = 0;
            for (int j = 0; j < n; ++j) sum += a[i*n+j]*x[j];
            ax[i] = sum;
            z[i] = std::min(std::max(sum+y[i], lo[i]), hi[i]);
            y[i] += sum-z[i];
            primal = std::max(primal, std::abs(sum-z[i]));
            norm_ax = std::max(norm_ax, std::abs(sum));
            norm_z = std::max(norm_z, std::abs(z[i]));
            work[i] = z[i]-old_z[i];
        }
        double dual = 0, norm_aty = 0;
        for (int i = 0; i < n; ++i) {
            double dz = 0, aty = 0;
            for (int j = 0; j < m; ++j) {
                dz += at[i*m+j]*work[j];
                aty += at[i*m+j]*y[j];
            }
            dual = std::max(dual, std::abs(rho*dz+sigma*(x[i]-old_x[i])));
            norm_aty = std::max(norm_aty, std::abs(aty));
        }
        const bool converged = primal <= abs_tol+rel_tol*std::max(norm_ax,norm_z)
                            && dual <= abs_tol+rel_tol*norm_aty;
        stats[0] = k; stats[1] = converged ? 1 : 0;
        if (converged) return k;
    }
    return max_iter;
}
