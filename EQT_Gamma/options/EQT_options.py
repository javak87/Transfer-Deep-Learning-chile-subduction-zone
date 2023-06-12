from options.base_options import BaseOptions


class EQTOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--base_model', type=str, default='instance', help='Load the base model EQT. check the existing model in seisbench github')
        parser.add_argument('--activate_check_point', type=bool, default=True, help='If you have trained EQT and have check point, you can assign True to use check point')
        parser.add_argument('--which_check_point', type=str, default='EQT_Trained_INSTANCE_CJN.pth.tar', help='Load the last trained model check point. please assign activate activate_check_point=True if you want to use check point. please put your check point in checkpoint folder')
        parser.add_argument('--P_threshold', type=float, default=0.2, help='P pick threshold to run EQT')
        parser.add_argument('--S_threshold', type=float, default=0.2, help='S pick threshold to run EQT')
        parser.add_argument('--detection_threshold', type=float, default=0.01, help='detection threshold to run EQT')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size to run pre-trained EQT. For more detail check the model.classify in seisbench github')
        parser.add_argument('----suffix_EQT', type=str, default='EQT_picks', help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        return parser


