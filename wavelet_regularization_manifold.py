import torch, numpy as np, matplotlib.pyplot as plt, geoopt


class Algo:
    class Manifold:
        def __init__(self, manifold:geoopt.Manifold, mid_point=lambda a,b,w:a+w*(b-a)):
            self.manifold = manifold
            self.mid_point = mid_point
            # self.mid_point = lambda a,b,w: manifold.expmap(a, a + w * manifold.logmap(a,b))
        def sub(self, a, b):
            return self.manifold.log(b, a)
        def dist(self, a, b):
            return self.manifold.dist(a, b)

    class Operator:
        def __init__(self, A):
            self.A = A
        def __mul__(self, other):
            return other @ self.A

    class Scheme():
        def __init__(self, mask, mid_point=lambda a,b,w:a + w * (b - a)):
            self.mask = mask
            self.mid_point = mid_point

        @property
        def supp(self):
            return torch.arange(len(self.mask))-len(self.mask)//2

        def __mul__(self, sig):
            def refine(sig, mask, mid_point):
                def interp(sig, mask, ii):
                    ret = sig[ii]
                    n = len(sig)
                    interp_mask = np.array(mask)
                    m = interp_mask.shape[0]
                    for jj in range(1, m // 2 + 1):
                        ret = mid_point(
                            mid_point(ret, sig[np.clip((ii + jj), 0, n - 1)],
                                      interp_mask[m // 2 + jj] / interp_mask[m // 2] / 2),
                            mid_point(ret, sig[np.clip((ii - jj), 0, n - 1)],
                                      interp_mask[m // 2 - jj] / interp_mask[m // 2] / 2),
                            0.5)
                        interp_mask[m // 2] += interp_mask[m // 2 + jj] + interp_mask[m // 2 - jj]
                    return ret

                n = sig.shape[0]
                ret = torch.zeros((2 * n, np.shape(sig)[1]))
                ret[::2, :] = sig
                for ii in range(n):
                    ret[ii * 2 + 1, :] = interp(sig, mask, ii)
                return ret
            return refine(sig, self.mask, self.mid_point)

    def __init__(self, u, S, R, manifold, A_op, coarse_lambda=1,multiscale_lambda=1,
                 p=2, q=2, mu=1, alpha=1, std_noise=2, lr=0.1):
        self.compare_now_to_start = None
        self.losses_details_history = []
        self.losses_coarsest_history = []
        self.losses_data_history = []
        self.lr = lr

        self.s = 1
        self.mu = mu
        self.R = R
        self.u = torch.tensor(u, requires_grad=True)
        self.f = torch.clone(self.u)

        self.S = S
        self.A_op = A_op
        self.coarse_lambda = coarse_lambda
        self.multiscale_lambda = multiscale_lambda
        self.std_noise = std_noise
        self.p = p
        self.q = q
        self.alpha = alpha

        if p == 0:
            self.title = r'$\mathcal{L}(u)=\sum_i dist(u_i,f_i)^2+' \
                     r'\lambda_{details} \cdot \sum_{n,r} 2^{2r} '\
                     r'\| d_{n,r}(u) \|^0_{\hat u_{n,r}}+'\
                     r'\lambda_{coarsest} \sum_{n,iter} dist(u_{n-e_i,0},u_{n,0})$'
        else:
            self.title = r'$\mathcal{L}(u)=\sum_i dist(u_i,f_i)^p + ' \
                     r'\lambda_{details} \cdot \sum_{n,r} 2^{rq(\alpha+\frac{1}{2}-\frac{1}{q})}'\
                     r'\| d_{n,r}(u) \|^q_{\hat u_{n,r}} + '\
                     r'\lambda_{coarsest} \sum_{n,iter} dist(u_{n-e_i,0},u_{n,0})^q$'


        self.manifold = manifold

        self.iter = 0
        self.losses_history = []
        self.fig = plt.figure(figsize=(26,13))
        if self.compare_now_to_start:
            self.ax_points = self.fig.add_subplot(121, projection='3d')
            self.ax_points.view_init(azim=-80, elev=45)
            self.ax_points2 = self.fig.add_subplot(122, projection='3d')
            self.ax_points2.view_init(azim=-80, elev=45)
            return

        self.ax_points = self.fig.add_subplot(131, projection='3d')
        self.ax_points.view_init(azim=-80, elev=45)

        self.ax_loss = self.fig.add_subplot(4,3,3)
        self.ax_loss_data = self.fig.add_subplot(4,3,6)
        self.ax_loss_coarsest = self.fig.add_subplot(4,3,9)
        self.ax_loss_details = self.fig.add_subplot(4, 3, 12)

        self.ax_wavelet = [self.fig.add_subplot(self.R+1, 3, 3 * r - 1) for r in range(1, self.R + 2)]


    def loss(self, parts=False):

        conv_u = torch.zeros(self.u.shape[0], 3)

        for axis in range(3):
            u_axis = self.u[:, axis]  # Extract the axis you want to convolve
            u_axis = u_axis.unsqueeze(0).unsqueeze(1)  # Add a channel dimension for convolution
            M_axis = self.A_op.unsqueeze(0).unsqueeze(1)  # Add batch and channel dimensions to the mask
            conv_result = torch.conv1d(u_axis.double(), M_axis.double())#[M_axis.shape[-1]//2:-M_axis.shape[-1]//2]
            conv_u[M_axis.shape[-1]//2:-M_axis.shape[-1]//2+1, axis] = conv_result.squeeze()

        loss_data = torch.mean(self.manifold.dist(conv_u, self.f)**self.p)
        loss_coarest = torch.mean(self.coarse_lambda * torch.mean(self.manifold.dist(torch.roll(self.u[::2 ** self.R], 1) , self.u[::2 ** self.R])**self.q))
        loss_details = torch.mean(self.multiscale_lambda*sum([2**(r*self.q*(self.alpha+0.5-1/self.q))*torch.mean(torch.norm(self.S * self.u[::2 ** (self.R - r + 1)] - self.u[::2 ** (self.R - r)])**self.q)
                                           for r in range(1, self.R+1)]))
        loss = loss_data + loss_coarest + loss_details

        if parts:
            return loss, loss_data, loss_coarest, loss_details
        else:
            return loss

    def optimize(self):
        self.iter += 1
        self.loss().backward()

        self.u.data = self.manifold.manifold.retr(self.u.data, -self.lr * self.u.grad.data)


    @staticmethod
    def interp(v1, v2, t):
        x1, y1, z1 = v1
        x2, y2, z2 = v2

        omega = torch.arccos(torch.clip(torch.dot(v1, v2), -1, 1))
        if omega == 0:
            return v1 * t + (1 - t) * v2

        v_slerp = v1 * torch.sin((1 - t) * omega) / torch.sin(omega) + \
                  v2 * torch.sin(t * omega) / torch.sin(omega)

        return v_slerp

    def visualize(self):
        ax = self.ax_points
        ax.cla()

        # ax.plot(*self.f.detach().numpy().T, 'o', color=(1, 0, 0, 0.15), label='f')
        ax.plot(*self.f.detach().numpy().T, 'p', color=(1, 0, 0, 0.25), label='noisy data')
        ax.plot(*self.clean_points.T, '-', color=(0, 0, 0, 0.4), label='clean data')
        ax.plot(*self.u.detach().numpy().T, 'o', color=(0, 0, 1, 0.25), label='u')
        # plt.legend()
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color=(1, 0, 0, 0.1))

        if self.compare_now_to_start:
            ax = self.ax_points2
            ax.cla()

            # ax.plot(*self.f.detach().numpy().T, 'o', color=(1, 0, 0, 0.15), label='f')
            # ax.plot(*self.f.detach().numpy().T, 'p', color=(1, 0, 0, 0.25), label='noisy data')
            ax.plot(*self.clean_points.T, '-', color=(0, 0, 0, 0.4), label='clean data')
            ax.plot(*self.u.detach().numpy().T, 'o', color=(0, 0, 1, 0.25), label='u')
            # plt.legend()
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            ax.plot_wireframe(x, y, z, color=(1, 0, 0, 0.1))
            plt.savefig("theory.png")
            print(self.iter)
            print(self.loss())

            plt.show(block=False)
            plt.pause(0.001)
            return
        mathcal_L = ' $\mathcal{L}=$'
        lam_det ='\\lambda_{details}'
        lam_cor ='\\lambda_{coarsest}'
        ax.set_title(
            f"$p$ : {self.p} | $q$ : {self.q} | $\\alpha$ : {self.alpha} | $\\sigma$ : {self.std_noise}\n"
            f"${lam_det}$ : {self.multiscale_lambda} | ${lam_cor}$ : {self.coarse_lambda} | \n"
            f"Iteration : {self.iter} | learning-rate : {self.lr:.2e} \n"
            f"{mathcal_L}{self.loss():0.3f} | "
            f"$ֿ\Delta SNR$ : {-10*np.log(np.sum((self.manifold.manifold.dist(self.f, torch.Tensor(self.clean_points))**2).detach().numpy())/np.sum((self.manifold.manifold.dist(self.f,self.u)**2).detach().numpy()))/np.log(10):.3f}",
            y=1.02)

        norms = [torch.norm(self.u[::2**self.R], dim=1).detach().numpy()]+[torch.norm(self.S*self.u[::2 ** (self.R - r+1)] - self.u[::2 ** (self.R - r)], dim=1).detach().numpy() for r in
                                              range(1, self.R+1)]

        for r in range(self.R+1):
            self.ax_wavelet[r].cla()

            self.ax_wavelet[r].scatter(np.arange(len(self.norms0[r])),
                                       self.norms0[r], edgecolors=(1,0,0,1), facecolors='none', linewidth=0.5)
            # Drawing vertical lines from points to the x-axis
            for x_val, y_val in zip(np.arange(len(self.norms0[r])), self.norms0[r]):
                self.ax_wavelet[r].vlines(x_val, 0, y_val, colors=(1,0,0,1), linestyles='dashed', linewidth=1, label='original wavelets')
            self.ax_wavelet[r].set_ylim(-0.01, 1.5)
            self.ax_wavelet[r].grid('on')
            self.ax_wavelet[r].set_title(r'$\| d_{}_{}(u) \|^2$'.format('n,',f'{r}'))

            self.ax_wavelet[r].scatter(np.arange(len(norms[r])),
                                       norms[r], edgecolors=(0,0,1,1), facecolors='none', linewidth=0.5)
            # Drawing vertical lines from points to the x-axis
            for x_val, y_val in zip(np.arange(len(norms[r])), norms[r]):
                self.ax_wavelet[r].vlines(x_val, 0, y_val, colors=(0,0,1,1), linestyles='dashed', linewidth=1, label='u wavelets')
            self.ax_wavelet[r].set_ylim(-0.01, 1.5)
            self.ax_wavelet[r].grid('on')
            self.ax_wavelet[r].set_title(r'$\| d_{}_{}(u) \|^2$'.format('n,',f'{r}'))


        if self.iter > 1:
            loss, loss_data, loss_coarsest, loss_details = self.loss(parts=True)

            self.losses_history.append(loss.detach().numpy())
            self.losses_data_history.append(loss_data.detach().numpy())
            self.losses_coarsest_history.append(loss_coarsest.detach().numpy())
            self.losses_details_history.append(loss_details.detach().numpy())

            self.ax_loss.cla()
            self.ax_loss.plot(np.arange(self.iter-1)/(self.iter-1), self.losses_history)
            loss_title = r'$\mathcal{L}$'
            self.ax_loss.set_title(f'{loss_title}[iter={self.iter}]')
            self.ax_loss.set_ylim(-0.01, 1.1*max(self.losses_history))
            self.ax_loss.set_xlim(-0.01, 1.01)

            self.ax_loss_data.cla()
            self.ax_loss_data.plot(np.arange(self.iter-1)/(self.iter-1), self.losses_data_history)
            loss_title = r'$\mathcal{L}_{data}$'
            self.ax_loss_data.set_title(f'{loss_title}[iter={self.iter}]')
            self.ax_loss_data.set_ylim(-0.01, 1.01*max(self.losses_data_history))
            self.ax_loss_data.set_xlim(-0.01, 1.01)

            self.ax_loss_coarsest.cla()
            self.ax_loss_coarsest.plot(np.arange(self.iter-1)/(self.iter-1), self.losses_coarsest_history)
            loss_title = r'$\mathcal{L}_{coarsest}$'
            self.ax_loss_coarsest.set_title(f'{loss_title}[iter={self.iter}]')
            self.ax_loss_coarsest.set_ylim(-0.01, 1.01*max(self.losses_history))
            self.ax_loss_coarsest.set_xlim(-0.01, 1.01)

            self.ax_loss_details.cla()
            self.ax_loss_details.plot(np.arange(self.iter-1)/(self.iter-1), self.losses_details_history)
            loss_title = r'$\mathcal{L}_{details}$'
            self.ax_loss_details.set_title(f'{loss_title}[iter={self.iter}]')
            self.ax_loss_details.set_ylim(-0.01, 1.01*max(self.losses_details_history))
            self.ax_loss_details.set_xlim(-0.01, 1.01)

        plt.suptitle(self.title)
        handles, labels = self.ax_loss.get_legend_handles_labels()
        handles_, labels_ = self.ax_wavelet[0].get_legend_handles_labels()
        handles+=handles_[-2:]
        labels+=labels_[-2:]
        handles_, labels_ = self.ax_points.get_legend_handles_labels()
        labels+=labels_
        handles+=handles_
        self.fig.legend(handles, labels, loc='lower left')

        plt.savefig("image2.png")
        plt.show(block=False)
        plt.pause(0.01)

    @staticmethod
    def run():
        manifold = Algo.Manifold(geoopt.manifolds.sphere.Sphere(), mid_point=Algo.interp)
        # manifold = Algo.Manifold(geoopt.manifolds.euclidean.Euclidean(), mid_point=Algo.interp)

        # manifold = Algo.Manifold(geoopt.manifolds.sphere.Sphere())
        # mask = Algo.Scheme(torch.Tensor([0.25,0.5,0.25]), mid_point=manifold.mid_point)
        mask = Algo.Scheme(torch.Tensor([-1/16,0,9/16,1,9/16,0,-1/16]), mid_point=manifold.mid_point)
        A_op = Algo.Operator(torch.eye(3))
        # get_p = lambda phi, theta: [np.cos(phi+np.random.randn()*0.01 * np.pi / 180) * np.cos(theta * np.pi / 180),
        #                             np.sin(phi * np.pi / 180) * np.cos(theta+np.random.randn()*0.01 * np.pi / 180),
        #                             np.sin(theta * np.pi / 180)]
        # points = torch.Tensor(np.array([get_p(theta*torch.pi, theta*4) for theta in np.linspace(0, 180, 128)]))

        num_points = 64#128
        sigma = 0.04
        # sigma = 0.4
        magnitude = 0.3#5
        omega = 16
        np.random.seed(42)
        t = np.linspace(-0.5, 0.5, num_points)
        f = lambda t: magnitude * np.sin(omega * t) * np.exp(-0.8 * t)
        g = lambda t: t#magnitude * np.cos(omega * t) * np.exp(-0.8 * t)+t**2
        clean_points = np.stack([
            g(t),
                                 f(t),
                                 np.clip((1 - g(t) ** 2 - f(t) ** 2), 0, 1) ** 0.5,
            # g(t)
        ], 1)
        t=3**2
        A_op = np.array([np.exp(-n**2/(2*t))*1/np.sqrt(2*t*np.pi) for n in range(-6,6)]).reshape(-1)
        A_op/=np.sum(A_op)
        # A_op = np.array([1])
        noisy_points=np.zeros_like(clean_points)
        for d in range(3):
            noisy_points[:,d] = np.convolve(clean_points[:,d], A_op, 'same') + np.random.randn(num_points, ) * sigma

        A_op = torch.Tensor(A_op)#np.stack([A_op,A_op,A_op],0))
        algo = Algo(noisy_points, mask, R=5, p=2, q=1, alpha=1, manifold=manifold, A_op=A_op,
                    coarse_lambda=1e-1, multiscale_lambda=1e-3,
                    std_noise=sigma, lr=1e-1)


        # # no coarse
        # algo = Algo(noisy_points, mask, R=3, p=2, q=1, alpha=1, manifold=manifold, A_op=A_op,
        #             coarse_lambda=0.0, multiscale_lambda=0.1,
        #             std_noise=sigma, lr=1e-3)
        #
        # # no multiscale
        # algo = Algo(noisy_points, mask, R=3, p=2, q=1, alpha=1, manifold=manifold, A_op=A_op,
        #             coarse_lambda=1.0, multiscale_lambda=0.0,
        #             std_noise=sigma, lr=1e-3)
        #
        # # no data
        # algo = Algo(noisy_points, mask, R=3, p=2, q=1, alpha=1, manifold=manifold,
        #             A_op=A_op, coarse_lambda=0.0, multiscale_lambda=100.0,
        #             std_noise=sigma, lr=1e-2)
        #
        # # no regularization
        # algo = Algo(noisy_points, mask, R=3, p=2, q=1, alpha=1, manifold=manifold, A_op=A_op,
        #             coarse_lambda=0.0, multiscale_lambda=0.0,
        #             std_noise=sigma, lr=1e-2)
        #
        #
        #
        # algo = Algo(noisy_points, mask, R=3, p=2, q=0.1, alpha=1.5, manifold=manifold, A_op=A_op,
        #             # coarse_lambda=0.00, multiscale_lambda=0.1,
        #             # coarse_lambda=0.04, multiscale_lambda=0.04,
        #             coarse_lambda=5e-2, multiscale_lambda=0.5,
        #             std_noise=sigma, lr=1e-1)
        #



        # algo = Algo(noisy_points, mask, R=3, p=2, q=1, alpha=1, manifold=manifold, A_op=A_op,
        #             coarse_lambda=5e-2, multiscale_lambda=5e-2,
        #             std_noise=sigma, lr=1e-1)

        # algo.visualize()
        algo.norms0 = [torch.norm(algo.u[::2**algo.R], dim=1).detach().numpy()]+[
            torch.norm(algo.S*algo.u[::2 ** (algo.R - r+1)] - algo.u[::2 ** (algo.R - r)], dim=1).detach().numpy() for r in
                                              range(1, algo.R+1)]
        algo.clean_points = clean_points
        algo.lr0 = algo.lr
        algo.decay =1# 0.95

        for ii in range(200):
            algo.lr = np.clip(algo.lr*algo.decay, algo.lr0/10,algo.lr0)
            if algo.lr == algo.lr0/10:
                algo.lr0 = algo.lr0 / 10
                algo.decay = 0.9+0.1*algo.decay
            algo.optimize()
            algo.visualize()
            # print(algo.u)
        # print(algo.details_d_n_r_at_scale_r(0))
if __name__ == '__main__':

    Algo.run()