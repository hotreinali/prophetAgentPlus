from abc import ABC, abstractmethod
from appium import webdriver  # For Appium integration
from appium.options.android import UiAutomator2Options
from appium.options.ios import XCUITestOptions
from droidbot.device_state import DeviceState  # Import existing DroidBot state
from .input_event import InputEvent, KeyEvent, IntentEvent, TouchEvent, ManualEvent, SetTextEvent, KillAppEvent

class AbstractDevice(ABC):
    @abstractmethod
    def get_current_state(self):
        """Return DeviceState with UI hierarchy."""
        pass

    @abstractmethod
    def send_event(self, event):
        """Send input event (e.g., tap, key)."""
        pass

    @abstractmethod
    def launch_app(self, app_path):
        """Launch app from .apk or .ipa."""
        pass

    @abstractmethod
    def get_top_activity(self):
        """Get current screen/activity name."""
        pass



class AppiumDevice(AbstractDevice):
    def __init__(self, platform='android', device_url=None, app_path=None, **caps):
        self.platform = platform.lower()
        self.driver = None
        self.desired_caps = caps
        self.desired_caps['app'] = app_path
        self.desired_caps['platformName'] = self.platform.title()
        self.device_url = device_url or 'http://localhost:4723'  # Default Appium server

    def connect(self):
        if self.platform == 'android':
            options = UiAutomator2Options().load_capabilities(self.desired_caps)
        elif self.platform == 'ios':
            options = XCUITestOptions().load_capabilities(self.desired_caps)
        else:
            raise ValueError("Unsupported platform")
        self.driver = webdriver.Remote(self.device_url, options=options)

    def disconnect(self):
        if self.driver:
            self.driver.quit()

    def get_current_state(self):
        if not self.driver:
            self.connect()
        # Get page source (XML hierarchy, similar to Android dumpsys)
        source = self.driver.page_source
        # Parse to DeviceState (reuse DroidBot's parser or use appium's element tree)
        views = self._parse_source_to_views(source)  # Implement XML parsing (e.g., via lxml)
        activity = self.driver.current_activity if self.platform == 'android' else self.driver.current_url or "iOS Page"
        state = DeviceState(views=views, foreground_activity=activity)
        return state

    def send_event(self, event):
        if isinstance(event, TouchEvent):
            el = self.driver.find_element(by=event.view['strategy'], value=event.view['identifier'])  # e.g., accessibility id
            el.click()
        elif isinstance(event, KeyEvent):
            self.driver.press_keycode(event.keycode)  # Map Android keys to iOS equivalents
        # Handle other events (SwipeEvent, etc.) similarly
        # For iOS-specific: Use driver.swipe() for gestures

    def launch_app(self, app_path):
        self.desired_caps['app'] = app_path
        self.connect()

    def get_top_activity(self):
        return self.driver.current_activity if self.platform == 'android' else self.driver.title

    def _parse_source_to_views(self, source):
        # Simple stub: Parse XML to list of dicts like {'text': , 'class': , 'bounds': }
        # Use xml.etree.ElementTree or lxml to extract elements
        import xml.etree.ElementTree as ET
        root = ET.fromstring(source)
        views = []
        for elem in root.iter():
            views.append({
                'text': elem.get('text'),
                'class': elem.get('class'),
                'resource-id': elem.get('resource-id') or elem.get('name'),  # Map iOS 'name' to Android 'resource-id'
                'bounds': elem.get('bounds') or f"[{elem.get('x')},{elem.get('y')},{elem.get('width')},{elem.get('height')}]"
            })
        return views