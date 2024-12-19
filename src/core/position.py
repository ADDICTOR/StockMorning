from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from ..utils.logger import setup_logger

class Position:
    """持仓信息"""
    def __init__(self, 
                 code: str,
                 name: str,
                 shares: int,
                 price: float):
        self.code = code
        self.name = name
        self.shares = shares
        self.price = price
        self.cost_basis = price
        self.market_value = shares * price
        
    def update_price(self, new_price: float):
        """更新价格"""
        self.price = new_price
        self.market_value = self.shares * new_price
        
    def add_shares(self, shares: int, price: float):
        """增加持仓"""
        # 计算新的成本基础（加权平均）
        total_cost = self.cost_basis * self.shares + price * shares
        self.shares += shares
        self.cost_basis = total_cost / self.shares if self.shares > 0 else 0
        self.price = price
        self.market_value = self.shares * price
        
    def reduce_shares(self, shares: int, price: float) -> float:
        """减少持仓，返回实现盈亏"""
        if shares > self.shares:
            return None
            
        # 计算实现盈亏
        realized_pnl = (price - self.cost_basis) * shares
        
        self.shares -= shares
        # 如果完全清仓，重置成本基础
        if self.shares == 0:
            self.cost_basis = 0
        self.price = price
        self.market_value = self.shares * price
        
        return realized_pnl
        
    def get_unrealized_pnl(self) -> float:
        """获取未实现盈亏"""
        return (self.price - self.cost_basis) * self.shares
        
    def get_position_value(self) -> float:
        """获取持仓市值"""
        return self.market_value
        
    def get_position_info(self) -> dict:
        """获取持仓信息"""
        return {
            'code': self.code,
            'name': self.name,
            'shares': self.shares,
            'price': self.price,
            'cost_basis': self.cost_basis,
            'market_value': self.market_value,
            'unrealized_pnl': self.get_unrealized_pnl(),
            'unrealized_pnl_pct': (self.price / self.cost_basis - 1) * 100 if self.cost_basis > 0 else 0
        }

class PositionManager:
    """持仓管理器"""
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.logger = setup_logger(__name__)
        
    def add_position(self, 
                    code: str,
                    name: str,
                    shares: int,
                    price: float) -> None:
        """添加持仓"""
        if code in self.positions:
            # 更新现有持仓
            pos = self.positions[code]
            total_cost = pos.cost_basis * pos.shares + price * shares
            total_shares = pos.shares + shares
            new_cost = total_cost / total_shares
            
            self.positions[code] = Position(
                code=code,
                name=name,
                shares=total_shares,
                price=new_cost
            )
        else:
            # 新建持仓
            self.positions[code] = Position(
                code=code,
                name=name,
                shares=shares,
                price=price
            )
            
    def reduce_position(self, 
                       code: str,
                       shares: int,
                       price: float) -> Optional[float]:
        """减少持仓"""
        if code not in self.positions:
            self.logger.error(f"未找到{code}的持仓")
            return None
            
        pos = self.positions[code]
        if shares > pos.shares:
            self.logger.error(f"{code}的卖出股数大于持仓股数")
            return None
            
        # 计算卖出收益
        realized_pnl = (price - pos.cost_basis) * shares
        
        # 更新持仓
        pos.shares -= shares
        if pos.shares == 0:
            del self.positions[code]
        else:
            pos.price = price
            pos.market_value = pos.shares * price
            pos.unrealized_pnl = (price - pos.cost_basis) * pos.shares
            
        return realized_pnl
        
    def update_prices(self, 
                     current_prices: Dict[str, float]) -> None:
        """更新持仓价格"""
        for code, price in current_prices.items():
            if code in self.positions:
                pos = self.positions[code]
                pos.price = price
                pos.market_value = pos.shares * price
                pos.unrealized_pnl = (price - pos.cost_basis) * pos.shares
                
    def get_position_summary(self) -> pd.DataFrame:
        """获取持仓摘要"""
        if not self.positions:
            return pd.DataFrame()
            
        summary = []
        for pos in self.positions.values():
            summary.append({
                '代码': pos.code,
                '名称': pos.name,
                '持仓数量': pos.shares,
                '成本价': pos.cost_basis,
                '现价': pos.price,
                '市值': pos.market_value,
                '浮动盈亏': pos.unrealized_pnl,
                '收益率': f"{pos.unrealized_pnl_pct:.2f}%"
            })
            
        return pd.DataFrame(summary) 