from .base_options import BaseOptions
from pyproj import CRS, Transformer


class GammaOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--proj_idt_from', type=int, default=4326, help= 'projection identifier that we want to use')
        parser.add_argument('--proj_idt_to', type=int, default=9155, help='projection identifier that we want to be projected')
        parser.add_argument('--use_dbscan', type=bool, default=True, help='If using dbscan to cut a long sequence of picks into segments. Using DBSCAN can significantly speed up associaiton using small windows.')
        parser.add_argument('--use_amplitude', type=bool, default=False, help='Use amplitude for association')
        
        parser.add_argument('--x_interval', type=tuple, default=(250, 600), help='The x interval of area for seismic association')
        parser.add_argument('--y_interval', type=tuple, default=(7200, 8000), help='The y interval of area for seismic association')
        parser.add_argument('--z_interval', type=tuple, default=(0, 150), help='The z interval of area for seismic association')

        parser.add_argument('--dbscan_eps', type=float, default=10, help='The maximum time between two picks for one to be considered as a neighbor of the other. See details in DBSCAN')
        parser.add_argument('--dbscan_min_samples', type=int, default=3, help='The number of samples in a neighborhood for a point to be considered as a core point. See details in DBSCAN')
        parser.add_argument('--method', type=str, default='BGMM', help='method should specify BGMM or GMM')


        parser.add_argument('--initial_points', type=list, default=[1,1,1], help='(default=[1,1,1] for (x, y, z) directions): Initial earthquake locations (cluster centers). For a large area over 10 degrees, more initial points are helpful, such as [2,2,1].')
        parser.add_argument('--covariance_prior', type=tuple, default=(5,5), help='(default = (5, 5)): covariance prior of time and amplitude residuals. Because current code only uses an uniform velocity model, a large covariance prior can be used to avoid splitting one event into multiple events.')
        
        # Filtering low quality association 
        parser.add_argument('--min_picks_per_eq', type=int, default=5, help='Minimum picks for associated earthquakes. We can also specify minimum P or S pick')
        parser.add_argument('--min_p_picks_per_eq', type=int, default=3, help='Minimum P-picks for associated earthquakes.')
        parser.add_argument('--min_s_picks_per_eq', type=int, default=3, help='Minimum S-picks for associated earthquakes.')
        parser.add_argument('--max_sigma11', type=float, default=2, help='Max phase time residual (s)')
        parser.add_argument('--max_sigma22', type=float, default=1, help='Max phase amplitude residual (in log scale)')
        parser.add_argument('--max_sigma12', type=float, default=1, help='Max covariance term. (Usually not used)')
        parser.add_argument('--pick_path', type=str, default='./result/picker_output/EQT_picks_bas_modl:instance_chk_pt:True_which_chk_pt:EQT_Trained_INSTANCE_CJN.pth.tar_p_th:0.2_s_th:0.2_det_th:0.01_batch_size:256.pkl', help='The path of generated picks by EQT')


        args=parser.parse_args()
        if args.method == 'BGMM':
            parser.add_argument('--oversampling_factor', type=float, default=4, help='The initial number of clusters is determined by (Number of picks)/(Number of stations)/(Inital points) * (oversampling factor).')
        
        else:
            parser.add_argument('--oversampling_factor', type=float, default=1, help='The initial number of clusters is determined by (Number of picks)/(Number of stations)/(Inital points) * (oversampling factor).')

        parser.add_argument('--transformer', type=float, default=Transformer.from_crs(CRS.from_epsg(args.proj_idt_from), CRS.from_epsg(args.proj_idt_to)), help='')
        return parser