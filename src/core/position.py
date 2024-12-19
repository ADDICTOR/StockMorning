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
                 cost: float,
                 current_price: float):
        self.code = code
        self.name = name
        self.shares = shares
        self.cost = cost
        self.current_price = current_price
        self.market_value = shares * current_price
        self.unrealized_pnl = (current_price - cost) * shares
        self.returns = (current_price - cost) / cost

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
            total_cost = pos.cost * pos.shares + price * shares
            total_shares = pos.shares + shares
            new_cost = total_cost / total_shares
            
            self.positions[code] = Position(
                code=code,
                name=name,
                shares=total_shares,
                cost=new_cost,
                current_price=price
            )
        else:
            # 新建持仓
            self.positions[code] = Position(
                code=code,
                name=name,
                shares=shares,
                cost=price,
                current_price=price
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
        realized_pnl = (price - pos.cost) * shares
        
        # 更新持仓
        pos.shares -= shares
        if pos.shares == 0:
            del self.positions[code]
        else:
            pos.current_price = price
            pos.market_value = pos.shares * price
            pos.unrealized_pnl = (price - pos.cost) * pos.shares
            
        return realized_pnl
        
    def update_prices(self, 
                     current_prices: Dict[str, float]) -> None:
        """更新持仓价格"""
        for code, price in current_prices.items():
            if code in self.positions:
                pos = self.positions[code]
                pos.current_price = price
                pos.market_value = pos.shares * price
                pos.unrealized_pnl = (price - pos.cost) * pos.shares
                pos.returns = (price - pos.cost) / pos.cost
                
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
                '成本价': pos.cost,
                '现价': pos.current_price,
                '市值': pos.market_value,
                '浮动盈亏': pos.unrealized_pnl,
                '收益率': f"{pos.returns*100:.2f}%"
            })
            
        return pd.DataFrame(summary) 