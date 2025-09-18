import streamlit as st
import hashlib
import json
import os
from pathlib import Path


class AuthManager:
    def __init__(self, auth_file="auth.json"):
        self.auth_file = Path(auth_file)
        self.secretary_password = (
            "admindarmalmed1432@"  # Change this to your desired secretary password
        )
        self.init_auth_file()

    def init_auth_file(self):
        """Initialize auth file if it doesn't exist (empty by default)"""
        if not self.auth_file.exists():
            # Start with empty users - no default admin
            default_users = {}
            self.save_users(default_users)

    def hash_password(self, password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def load_users(self):
        """Load users from auth file"""
        try:
            with open(self.auth_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_users(self, users):
        """Save users to auth file"""
        with open(self.auth_file, "w") as f:
            json.dump(users, f, indent=2)

    def verify_user(self, username, password):
        """Verify user credentials"""
        users = self.load_users()
        hashed_password = self.hash_password(password)
        return users.get(username) == hashed_password

    def change_password(self, username, new_password):
        """Change user password"""
        users = self.load_users()
        if username not in users:
            return False  # User doesn't exist
        users[username] = self.hash_password(new_password)
        self.save_users(users)
        return True

    def add_user(self, username, password):
        """Add new user"""
        users = self.load_users()
        if username in users:
            return False  # User already exists
        users[username] = self.hash_password(password)
        self.save_users(users)
        return True

    def delete_user(self, username):
        """Delete user"""
        users = self.load_users()
        if username not in users:
            return False  # User doesn't exist
        del users[username]
        self.save_users(users)
        return True

    def get_user_list(self):
        """Get list of all usernames"""
        users = self.load_users()
        return list(users.keys())

    def verify_secretary_password(self, password):
        """Verify secretary password for admin functions"""
        return password == self.secretary_password

    def has_users(self):
        """Check if any users exist"""
        users = self.load_users()
        return len(users) > 0


def show_login_page():
    """Display login page"""
    st.set_page_config(page_title="股票分析系統 - 登入", layout="centered")

    st.title("🔐 股票分析系統")
    st.markdown("---")

    auth_manager = AuthManager()

    # Check if any users exist
    if not auth_manager.has_users():
        # No users exist - show only user creation interface
        st.warning("🚫 系統中未找到使用者")
        st.info("👤 請使用管理員密碼建立第一個使用者帳號")

        st.subheader("🔧 建立第一個使用者")
        st.info("🔑 需要管理員密碼才能建立使用者")

        with st.form("first_user_creation"):
            secretary_pwd = st.text_input(
                "管理員密碼",
                type="password",
                placeholder="請輸入管理員密碼",
            )

            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input(
                    "新使用者名稱", placeholder="請輸入使用者名稱"
                )
            with col2:
                new_password = st.text_input(
                    "新密碼", type="password", placeholder="請輸入密碼"
                )

            submit_first_user = st.form_submit_button(
                "建立第一個使用者", use_container_width=True
            )

            if submit_first_user:
                if not secretary_pwd:
                    st.warning("⚠️ 需要管理員密碼")
                elif not auth_manager.verify_secretary_password(secretary_pwd):
                    st.error("❌ 管理員密碼錯誤")
                elif not new_username or not new_password:
                    st.warning("⚠️ 請填寫所有欄位")
                else:
                    if auth_manager.add_user(new_username, new_password):
                        st.success(f"✅ 第一個使用者已建立：{new_username}")
                        st.info("請重新整理頁面以登入")
                        st.balloons()
                    else:
                        st.error(f"❌ 建立使用者失敗：{new_username}")

        st.stop()  # Stop here if no users exist

    # Create tabs for login and password management (only if users exist)
    tab1, tab2 = st.tabs(["🔑 登入", "🔧 使用者管理"])

    with tab1:
        st.subheader("請登入以繼續")

        with st.form("login_form"):
            username = st.text_input("使用者名稱", placeholder="請輸入使用者名稱")
            password = st.text_input("密碼", type="password", placeholder="請輸入密碼")
            submit_button = st.form_submit_button("登入", use_container_width=True)

            if submit_button:
                if username and password:
                    if auth_manager.verify_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("登入成功！正在重新導向...")
                        st.rerun()
                    else:
                        st.error("❌ 使用者名稱或密碼錯誤")
                else:
                    st.warning("⚠️ 請輸入使用者名稱和密碼")

        st.info(f"💡 如需帳號存取權限，請聯絡您的管理員")

    with tab2:
        st.subheader("使用者管理")
        st.info("🔑 需要管理員密碼才能管理使用者")

        # Show current users (for reference)
        with st.expander("👥 目前使用者"):
            current_users = auth_manager.get_user_list()
            if current_users:
                for i, user in enumerate(current_users, 1):
                    st.write(f"{i}. **{user}**")
            else:
                st.write("未找到使用者")

        # Move action selection OUTSIDE the form
        action = st.radio("操作", ["新增使用者", "變更密碼", "刪除使用者"])

        # Now create the form based on the selected action
        with st.form("user_management"):
            secretary_pwd = st.text_input(
                "管理員密碼",
                type="password",
                placeholder="請輸入管理員密碼",
            )

            # Dynamic interface based on action selection
            if action == "新增使用者":
                col1, col2 = st.columns(2)
                with col1:
                    target_username = st.text_input(
                        "新使用者名稱", placeholder="請輸入使用者名稱"
                    )
                with col2:
                    new_password = st.text_input(
                        "密碼", type="password", placeholder="請輸入密碼"
                    )

            elif action == "變更密碼":
                col1, col2 = st.columns(2)
                with col1:
                    target_username = st.text_input(
                        "使用者名稱", placeholder="請輸入要變更密碼的使用者名稱"
                    )
                with col2:
                    new_password = st.text_input(
                        "新密碼", type="password", placeholder="請輸入新密碼"
                    )

            elif action == "刪除使用者":
                # Show selectbox immediately for delete action
                current_users = auth_manager.get_user_list()
                if current_users:
                    col1, col2 = st.columns(2)
                    with col1:
                        target_username = st.selectbox(
                            "選擇要刪除的使用者",
                            options=["-- 請選擇使用者 --"] + current_users,
                            help="選擇您要刪除的使用者",
                        )
                        # Handle placeholder selection
                        if target_username == "-- 請選擇使用者 --":
                            target_username = None

                    with col2:
                        st.write("")  # Empty space for layout alignment
                        if target_username and target_username != "-- 請選擇使用者 --":
                            st.error(f"⚠️ 將要刪除使用者：**{target_username}**")
                        st.warning("⚠️ 此操作無法復原！")
                        st.info("ℹ️ 只需要管理員密碼和選擇使用者名稱")
                else:
                    st.warning("❌ 沒有可刪除的使用者")
                    target_username = None

                # Set new_password to None for delete action
                new_password = None

            submit_action = st.form_submit_button("執行", use_container_width=True)

            if submit_action:
                if not secretary_pwd:
                    st.warning("⚠️ 需要管理員密碼")
                elif not auth_manager.verify_secretary_password(secretary_pwd):
                    st.error("❌ 管理員密碼錯誤")
                elif not target_username:
                    st.warning("⚠️ 請選擇/輸入使用者名稱")
                else:
                    try:
                        if action == "新增使用者":
                            if not new_password:
                                st.warning("⚠️ 請輸入密碼")
                            elif auth_manager.add_user(target_username, new_password):
                                st.success(f"✅ 新使用者已新增：{target_username}")
                            else:
                                st.error(f"❌ 使用者 '{target_username}' 已存在")

                        elif action == "變更密碼":
                            if not new_password:
                                st.warning("⚠️ 請輸入新密碼")
                            elif auth_manager.change_password(
                                target_username, new_password
                            ):
                                st.success(f"✅ 使用者 {target_username} 的密碼已變更")
                            else:
                                st.error(f"❌ 找不到使用者 '{target_username}'")

                        elif action == "刪除使用者":
                            # Check if this is the last user
                            current_users = auth_manager.get_user_list()
                            if len(current_users) <= 1:
                                st.error(
                                    "❌ 無法刪除最後一個使用者！系統必須至少保留一個使用者。"
                                )
                            elif auth_manager.delete_user(target_username):
                                st.success(f"✅ 使用者已刪除：{target_username}")
                                st.info("使用者清單已更新。請重新整理頁面查看變更。")
                            else:
                                st.error(f"❌ 刪除使用者失敗：{target_username}")

                    except Exception as e:
                        st.error(f"❌ 錯誤：{str(e)}")


def check_authentication():
    """Check if user is authenticated"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        show_login_page()
        st.stop()


def show_logout_button():
    """Show logout button in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**👤 目前登入使用者：** {st.session_state.get('username', '未知')}"
    )

    if st.sidebar.button("🚪 登出", use_container_width=True):
        # Clear session state
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
