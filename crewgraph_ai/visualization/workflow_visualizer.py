"""
Workflow Visualizer - Comprehensive workflow visualization system

Provides graph plotting with networkx/matplotlib, interactive HTML visualizations
with plotly, workflow topology display, node status and execution flow.
"""

import os
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError

logger = get_logger(__name__)


class WorkflowVisualizer:
    """
    Comprehensive workflow visualization system for CrewGraph AI.
    
    Provides multiple visualization formats including static graphs with matplotlib,
    interactive HTML visualizations with plotly, and various export formats.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the WorkflowVisualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
            
        Raises:
            CrewGraphError: If visualization dependencies are not available
        """
        if not VISUALIZATION_AVAILABLE:
            raise CrewGraphError(
                "Visualization dependencies not available. "
                "Install with: pip install crewgraph-ai[visualization]"
            )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color schemes for different visualization modes
        self.colors = {
            'node_default': '#87CEEB',
            'node_running': '#FFD700', 
            'node_completed': '#90EE90',
            'node_failed': '#FFB6C1',
            'node_pending': '#D3D3D3',
            'edge_default': '#808080',
            'edge_active': '#FF4500',
            'text': '#000000'
        }
        
        logger.info(f"WorkflowVisualizer initialized with output directory: {output_dir}")
    
    def visualize_workflow_graph(self, 
                                workflow_data: Dict[str, Any],
                                title: str = "CrewGraph Workflow",
                                format: str = "html",
                                show_details: bool = True) -> str:
        """
        Generate visual representation of workflow graph.
        
        Args:
            workflow_data: Workflow structure data containing nodes and edges
            title: Title for the visualization
            format: Output format ('html', 'png', 'svg', 'pdf')
            show_details: Whether to show detailed node information
            
        Returns:
            Path to the generated visualization file
            
        Raises:
            CrewGraphError: If visualization generation fails
            
        Example:
            ```python
            visualizer = WorkflowVisualizer()
            workflow_data = {
                'nodes': [
                    {'id': 'task1', 'name': 'Data Collection', 'status': 'completed'},
                    {'id': 'task2', 'name': 'Analysis', 'status': 'running'}
                ],
                'edges': [
                    {'source': 'task1', 'target': 'task2', 'type': 'dependency'}
                ]
            }
            path = visualizer.visualize_workflow_graph(workflow_data)
            ```
        """
        try:
            if format.lower() == 'html':
                return self._create_interactive_graph(workflow_data, title, show_details)
            else:
                return self._create_static_graph(workflow_data, title, format, show_details)
                
        except Exception as e:
            logger.error(f"Failed to generate workflow visualization: {e}")
            raise CrewGraphError(f"Visualization generation failed: {e}")
    
    def _create_interactive_graph(self, 
                                 workflow_data: Dict[str, Any],
                                 title: str,
                                 show_details: bool) -> str:
        """Create interactive HTML visualization using plotly."""
        # Create networkx graph for layout
        G = nx.DiGraph()
        
        # Add nodes
        nodes = workflow_data.get('nodes', [])
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # Add edges
        edges = workflow_data.get('edges', [])
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Prepare plotly traces
        edge_traces = []
        node_trace = self._create_node_trace(G, pos, show_details)
        
        # Create edge traces
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color=self.colors['edge_default']),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        
        fig.update_layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="CrewGraph AI Workflow Visualization",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(color="#999", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        # Save interactive plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_interactive_{timestamp}.html"
        filepath = self.output_dir / filename
        
        pyo.plot(fig, filename=str(filepath), auto_open=False)
        
        logger.info(f"Interactive workflow visualization saved to: {filepath}")
        return str(filepath)
    
    def _create_node_trace(self, G: nx.DiGraph, pos: Dict, show_details: bool) -> go.Scatter:
        """Create plotly node trace with hover information."""
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        hover_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node data
            node_data = G.nodes[node]
            status = node_data.get('status', 'pending')
            name = node_data.get('name', node)
            
            # Set color based on status
            color = self.colors.get(f'node_{status}', self.colors['node_default'])
            node_colors.append(color)
            
            # Node text
            node_text.append(name[:15] + "..." if len(name) > 15 else name)
            
            # Hover information
            if show_details:
                hover_info = f"<b>{name}</b><br>"
                hover_info += f"ID: {node}<br>"
                hover_info += f"Status: {status}<br>"
                
                # Add additional node data
                for key, value in node_data.items():
                    if key not in ['name', 'status', 'id']:
                        hover_info += f"{key}: {value}<br>"
                        
                hover_text.append(hover_info)
            else:
                hover_text.append(f"<b>{name}</b><br>Status: {status}")
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=30,
                color=node_colors,
                line=dict(width=2, color='black')
            ),
            text=node_text,
            textposition="middle center",
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            textfont=dict(size=10, color='black')
        )
    
    def _create_static_graph(self, 
                           workflow_data: Dict[str, Any],
                           title: str,
                           format: str,
                           show_details: bool) -> str:
        """Create static visualization using matplotlib."""
        plt.figure(figsize=(12, 8))
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        nodes = workflow_data.get('nodes', [])
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # Add edges
        edges = workflow_data.get('edges', [])
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color=self.colors['edge_default'], 
                              arrows=True, arrowsize=20, width=2)
        
        # Draw nodes with status-based colors
        node_colors = []
        for node in G.nodes():
            node_data = G.nodes[node]
            status = node_data.get('status', 'pending')
            color = self.colors.get(f'node_{status}', self.colors['node_default'])
            node_colors.append(color)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=2000, alpha=0.9)
        
        # Draw labels
        labels = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            name = node_data.get('name', node)
            labels[node] = name[:10] + "..." if len(name) > 10 else name
            
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        plt.title(title, size=16, weight='bold')
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.colors['node_pending'], label='Pending'),
            patches.Patch(color=self.colors['node_running'], label='Running'),
            patches.Patch(color=self.colors['node_completed'], label='Completed'),
            patches.Patch(color=self.colors['node_failed'], label='Failed')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save static plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_static_{timestamp}.{format.lower()}"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, format=format.lower(), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Static workflow visualization saved to: {filepath}")
        return str(filepath)
    
    def create_execution_timeline(self, 
                                 execution_data: List[Dict[str, Any]],
                                 title: str = "Workflow Execution Timeline") -> str:
        """
        Create timeline visualization of workflow execution.
        
        Args:
            execution_data: List of execution events with timestamps
            title: Title for the timeline
            
        Returns:
            Path to the generated timeline file
        """
        try:
            # Convert execution data to DataFrame
            df = pd.DataFrame(execution_data)
            
            if df.empty:
                raise CrewGraphError("No execution data provided for timeline")
            
            # Create timeline figure
            fig = go.Figure()
            
            # Group by task/node
            for task in df['task'].unique():
                task_data = df[df['task'] == task]
                
                fig.add_trace(go.Scatter(
                    x=task_data['timestamp'],
                    y=[task] * len(task_data),
                    mode='markers+lines',
                    name=task,
                    marker=dict(size=8),
                    hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Event: %{text}<extra></extra>',
                    text=task_data['event']
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Tasks",
                hovermode='closest',
                height=400 + len(df['task'].unique()) * 50
            )
            
            # Save timeline
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"execution_timeline_{timestamp}.html"
            filepath = self.output_dir / filename
            
            pyo.plot(fig, filename=str(filepath), auto_open=False)
            
            logger.info(f"Execution timeline saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create execution timeline: {e}")
            raise CrewGraphError(f"Timeline generation failed: {e}")
    
    def generate_workflow_summary(self, 
                                 workflow_data: Dict[str, Any],
                                 execution_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive workflow summary with statistics.
        
        Args:
            workflow_data: Workflow structure data
            execution_stats: Optional execution statistics
            
        Returns:
            Dictionary containing workflow summary and metrics
        """
        try:
            summary = {
                'workflow_info': {
                    'total_nodes': len(workflow_data.get('nodes', [])),
                    'total_edges': len(workflow_data.get('edges', [])),
                    'generation_time': datetime.now().isoformat()
                }
            }
            
            # Node status distribution
            nodes = workflow_data.get('nodes', [])
            status_counts = {}
            for node in nodes:
                status = node.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            summary['node_status_distribution'] = status_counts
            
            # Graph connectivity metrics
            if nodes:
                G = nx.DiGraph()
                for node in nodes:
                    G.add_node(node['id'])
                for edge in workflow_data.get('edges', []):
                    G.add_edge(edge['source'], edge['target'])
                
                summary['graph_metrics'] = {
                    'is_connected': nx.is_weakly_connected(G),
                    'number_of_components': nx.number_weakly_connected_components(G),
                    'average_clustering': nx.average_clustering(G.to_undirected()),
                    'density': nx.density(G)
                }
            
            # Add execution statistics if provided
            if execution_stats:
                summary['execution_stats'] = execution_stats
            
            # Save summary to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workflow_summary_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Workflow summary saved to: {filepath}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate workflow summary: {e}")
            raise CrewGraphError(f"Summary generation failed: {e}")
    
    def export_to_graphviz(self, 
                          workflow_data: Dict[str, Any],
                          filename: Optional[str] = None) -> str:
        """
        Export workflow to Graphviz DOT format.
        
        Args:
            workflow_data: Workflow structure data
            filename: Optional custom filename
            
        Returns:
            Path to the generated DOT file
        """
        try:
            from graphviz import Digraph
            
            dot = Digraph(comment='CrewGraph Workflow')
            dot.attr(rankdir='TB')
            
            # Add nodes
            nodes = workflow_data.get('nodes', [])
            for node in nodes:
                node_id = node['id']
                name = node.get('name', node_id)
                status = node.get('status', 'pending')
                
                # Set node style based on status
                if status == 'completed':
                    dot.node(node_id, name, style='filled', fillcolor='lightgreen')
                elif status == 'running':
                    dot.node(node_id, name, style='filled', fillcolor='yellow')
                elif status == 'failed':
                    dot.node(node_id, name, style='filled', fillcolor='lightcoral')
                else:
                    dot.node(node_id, name, style='filled', fillcolor='lightgray')
            
            # Add edges
            edges = workflow_data.get('edges', [])
            for edge in edges:
                dot.edge(edge['source'], edge['target'])
            
            # Save DOT file
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"workflow_graphviz_{timestamp}"
            
            filepath = self.output_dir / f"{filename}.dot"
            
            with open(filepath, 'w') as f:
                f.write(dot.source)
            
            logger.info(f"Graphviz DOT file saved to: {filepath}")
            return str(filepath)
            
        except ImportError:
            raise CrewGraphError("Graphviz not available. Install with: pip install graphviz")
        except Exception as e:
            logger.error(f"Failed to export to Graphviz: {e}")
            raise CrewGraphError(f"Graphviz export failed: {e}")