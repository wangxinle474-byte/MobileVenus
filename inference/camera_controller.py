"""
相机控制器
负责执行参数调整和与相机 API 交互
"""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class CameraMode(Enum):
    """相机模式"""
    AUTO = "auto"
    PORTRAIT = "portrait"
    NIGHT = "night"
    HDR = "hdr"
    PANORAMA = "panorama"


class FlashMode(Enum):
    """闪光灯模式"""
    OFF = "off"
    ON = "on"
    AUTO = "auto"


@dataclass
class CameraParameters:
    """相机参数数据类"""
    ev_compensation: float = 0.0        # 曝光补偿 [-2, 2]
    iso: int = 200                      # ISO [100, 3200]
    white_balance: int = 5500           # 色温 [2500, 9000]
    zoom_factor: float = 1.0            # 焦距 [0.5, 10]
    focus_point: tuple = (0.5, 0.5)     # 对焦点 (x, y)
    mode: CameraMode = CameraMode.AUTO
    flash: FlashMode = FlashMode.OFF
    shutter_speed: Optional[float] = None  # 快门速度（秒）
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'ev_compensation': self.ev_compensation,
            'iso': self.iso,
            'white_balance': self.white_balance,
            'zoom_factor': self.zoom_factor,
            'focus_point': self.focus_point,
            'mode': self.mode.value,
            'flash': self.flash.value,
            'shutter_speed': self.shutter_speed
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CameraParameters':
        """从字典创建"""
        params = cls()
        
        if 'ev_compensation' in data:
            params.ev_compensation = data['ev_compensation']
        if 'iso' in data:
            params.iso = int(data['iso'])
        if 'white_balance' in data:
            params.white_balance = int(data['white_balance'])
        if 'zoom_factor' in data:
            params.zoom_factor = data['zoom_factor']
        if 'focus_point' in data:
            params.focus_point = tuple(data['focus_point'])
        if 'mode' in data:
            params.mode = CameraMode(data['mode'])
        if 'flash' in data:
            params.flash = FlashMode(data['flash'])
        if 'shutter_speed' in data:
            params.shutter_speed = data['shutter_speed']
        
        return params


class CameraController:
    """
    相机控制器
    负责执行参数调整和与相机 API 交互
    """
    
    def __init__(self, platform: str = 'ios'):
        """
        Args:
            platform: 平台 ('ios' 或 'android')
        """
        self.platform = platform
        self.current_params = CameraParameters()
        self.callbacks = {
            'on_param_change': [],
            'on_error': []
        }
        
    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """触发回调"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                callback(*args, **kwargs)
    
    def apply_parameters(
        self,
        params: Dict,
        smooth: bool = True,
        validate: bool = True
    ) -> bool:
        """
        应用参数到相机
        
        Args:
            params: 参数字典
            smooth: 是否平滑过渡
            validate: 是否验证参数
            
        Returns:
            success: 是否成功
        """
        try:
            # 转换为 CameraParameters 对象
            new_params = CameraParameters.from_dict(params)
            
            # 应用各个参数
            if 'ev_compensation' in params:
                self._set_ev_compensation(new_params.ev_compensation)
            
            if 'iso' in params:
                self._set_iso(new_params.iso)
            
            if 'white_balance' in params:
                self._set_white_balance(new_params.white_balance)
            
            if 'zoom_factor' in params:
                self._set_zoom(new_params.zoom_factor)
            
            if 'focus_point' in params:
                self._set_focus_point(new_params.focus_point)
            
            if 'mode' in params:
                self._set_mode(new_params.mode)
            
            if 'flash' in params:
                self._set_flash(new_params.flash)
            
            if 'shutter_speed' in params:
                self._set_shutter_speed(new_params.shutter_speed)
            
            # 更新当前参数
            self.current_params = new_params
            
            # 触发回调
            self._trigger_callback('on_param_change', new_params)
            
            return True
            
        except Exception as e:
            self._trigger_callback('on_error', str(e))
            return False
    
    def _set_ev_compensation(self, value: float):
        """设置曝光补偿"""
        if self.platform == 'ios':
            # iOS AVCaptureDevice API
            # device.setExposureTargetBias(value, completionHandler: nil)
            print(f"[iOS] 设置 EV 补偿: {value}")
        else:
            # Android Camera2 API
            # captureRequestBuilder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, value)
            print(f"[Android] 设置 EV 补偿: {value}")
    
    def _set_iso(self, value: int):
        """设置 ISO"""
        if self.platform == 'ios':
            # device.setExposureModeCustom(duration: CMTime, iso: Float, completionHandler: nil)
            print(f"[iOS] 设置 ISO: {value}")
        else:
            # captureRequestBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, value)
            print(f"[Android] 设置 ISO: {value}")
    
    def _set_white_balance(self, value: int):
        """设置白平衡"""
        if self.platform == 'ios':
            # device.setWhiteBalanceModeLocked(with: gains, completionHandler: nil)
            print(f"[iOS] 设置白平衡: {value}K")
        else:
            # captureRequestBuilder.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_OFF)
            print(f"[Android] 设置白平衡: {value}K")
    
    def _set_zoom(self, value: float):
        """设置焦距"""
        if self.platform == 'ios':
            # device.videoZoomFactor = value
            print(f"[iOS] 设置焦距: {value}x")
        else:
            # captureRequestBuilder.set(CaptureRequest.SCALER_CROP_REGION, rect)
            print(f"[Android] 设置焦距: {value}x")
    
    def _set_focus_point(self, point: tuple):
        """设置对焦点"""
        x, y = point
        if self.platform == 'ios':
            # device.setFocusModeLocked(lensPosition: position, completionHandler: nil)
            print(f"[iOS] 设置对焦点: ({x:.2f}, {y:.2f})")
        else:
            # captureRequestBuilder.set(CaptureRequest.CONTROL_AF_REGIONS, regions)
            print(f"[Android] 设置对焦点: ({x:.2f}, {y:.2f})")
    
    def _set_mode(self, mode: CameraMode):
        """设置拍摄模式"""
        print(f"[{self.platform}] 设置模式: {mode.value}")
        
        # 不同模式可能需要组合多个参数
        if mode == CameraMode.PORTRAIT:
            # 启用人像模式（景深效果）
            pass
        elif mode == CameraMode.NIGHT:
            # 夜景模式（长曝光）
            pass
        elif mode == CameraMode.HDR:
            # HDR 模式
            pass
    
    def _set_flash(self, flash: FlashMode):
        """设置闪光灯"""
        if self.platform == 'ios':
            # device.flashMode = .on/.off/.auto
            print(f"[iOS] 设置闪光灯: {flash.value}")
        else:
            # captureRequestBuilder.set(CaptureRequest.FLASH_MODE, mode)
            print(f"[Android] 设置闪光灯: {flash.value}")
    
    def _set_shutter_speed(self, value: Optional[float]):
        """设置快门速度"""
        if value is None:
            return
        
        if self.platform == 'ios':
            # device.setExposureModeCustom(duration: CMTime, iso: Float, completionHandler: nil)
            print(f"[iOS] 设置快门速度: 1/{int(1/value)}s")
        else:
            # captureRequestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, nanoseconds)
            print(f"[Android] 设置快门速度: 1/{int(1/value)}s")
    
    def get_current_parameters(self) -> CameraParameters:
        """获取当前参数"""
        return self.current_params
    
    def reset_to_auto(self):
        """重置为自动模式"""
        auto_params = CameraParameters()
        self.apply_parameters(auto_params.to_dict())


class SmartCameraController(CameraController):
    """
    智能相机控制器
    集成参数预测和自动调整
    """
    
    def __init__(
        self,
        platform: str = 'ios',
        auto_adjust: bool = True,
        user_confirmation: bool = False
    ):
        """
        Args:
            platform: 平台
            auto_adjust: 是否自动调整参数
            user_confirmation: 是否需要用户确认
        """
        super().__init__(platform)
        
        self.auto_adjust = auto_adjust
        self.user_confirmation = user_confirmation
        self.pending_params = None
        
    def suggest_parameters(
        self,
        predicted_params: Dict,
        problem_description: str
    ) -> Dict:
        """
        建议参数调整
        
        Args:
            predicted_params: 预测的参数
            problem_description: 问题描述
            
        Returns:
            suggestion: 建议信息
        """
        suggestion = {
            'params': predicted_params,
            'description': problem_description,
            'changes': self._get_parameter_changes(predicted_params),
            'auto_apply': self.auto_adjust and not self.user_confirmation
        }
        
        if suggestion['auto_apply']:
            # 自动应用
            self.apply_parameters(predicted_params)
        else:
            # 等待用户确认
            self.pending_params = predicted_params
        
        return suggestion
    
    def _get_parameter_changes(self, new_params: Dict) -> List[Dict]:
        """获取参数变化列表"""
        changes = []
        current = self.current_params.to_dict()
        
        for key, new_value in new_params.items():
            if key in current:
                old_value = current[key]
                if old_value != new_value:
                    changes.append({
                        'parameter': key,
                        'old_value': old_value,
                        'new_value': new_value,
                        'description': self._describe_change(key, old_value, new_value)
                    })
        
        return changes
    
    def _describe_change(self, param: str, old_value, new_value) -> str:
        """描述参数变化"""
        descriptions = {
            'ev_compensation': f"曝光补偿: {old_value:+.1f} → {new_value:+.1f}",
            'iso': f"ISO: {old_value} → {new_value}",
            'white_balance': f"白平衡: {old_value}K → {new_value}K",
            'zoom_factor': f"焦距: {old_value}x → {new_value}x",
            'mode': f"模式: {old_value} → {new_value}",
            'flash': f"闪光灯: {old_value} → {new_value}"
        }
        
        return descriptions.get(param, f"{param}: {old_value} → {new_value}")
    
    def confirm_and_apply(self, apply: bool = True):
        """确认并应用待处理的参数"""
        if self.pending_params is None:
            return False
        
        if apply:
            success = self.apply_parameters(self.pending_params)
            self.pending_params = None
            return success
        else:
            self.pending_params = None
            return False


if __name__ == "__main__":
    # 测试相机控制器
    print("=" * 50)
    print("测试相机控制器")
    print("=" * 50)
    
    # 创建控制器
    controller = SmartCameraController(platform='ios', auto_adjust=False)
    
    # 注册回调
    def on_param_change(params):
        print(f"\n✅ 参数已更新: {params.to_dict()}")
    
    def on_error(error):
        print(f"\n❌ 错误: {error}")
    
    controller.register_callback('on_param_change', on_param_change)
    controller.register_callback('on_error', on_error)
    
    # 测试参数建议
    print("\n场景: 检测到光线不足")
    predicted_params = {
        'ev_compensation': 1.5,
        'iso': 400,
        'mode': 'hdr'
    }
    
    suggestion = controller.suggest_parameters(
        predicted_params,
        "光线不足，主体欠曝"
    )
    
    print(f"\n建议:")
    print(f"  问题: {suggestion['description']}")
    print(f"  参数变化:")
    for change in suggestion['changes']:
        print(f"    - {change['description']}")
    
    # 用户确认并应用
    print("\n用户确认应用...")
    controller.confirm_and_apply(apply=True)
    
    # 查看当前参数
    current = controller.get_current_parameters()
    print(f"\n当前参数: {current.to_dict()}")
