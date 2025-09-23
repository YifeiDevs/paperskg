#!/usr/bin/env python3
"""
PapersKG MCP Server 核心功能测试

测试三个核心功能：
1. index_folder - 索引PDF文件
2. ingest_paper - 摄取论文文本  
3. extract_entities - 提取材料实体

运行方法：
    python test_mcp_client.py
"""

import logging
import pathlib
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 导入项目模块
try:
    from schemas import Material
    import server
    from server import papers_index
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)


def test_index_folder():
    """测试文件夹索引功能"""
    logger.info("=== 测试 index_folder 功能 ===")
    
    papers_dir = "./papers"
    
    # 检查papers目录是否存在PDF文件
    papers_path = pathlib.Path(papers_dir)
    if not papers_path.exists():
        logger.warning(f"papers目录不存在: {papers_dir}")
        return False
        
    pdf_files = list(papers_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"papers目录中没有PDF文件: {papers_dir}")
        return False
    
    logger.info(f"发现 {len(pdf_files)} 个PDF文件")
    
    # 清空papers_index
    papers_index.clear()
    
    # 测试索引功能
    try:
        result = server.index_folder(papers_dir)
        
        # 验证结果
        assert isinstance(result, list), "返回结果应该是列表"
        assert len(result) == len(pdf_files), f"应该索引{len(pdf_files)}个文件，实际索引了{len(result)}个"
        assert len(papers_index) == len(pdf_files), f"papers_index应该有{len(pdf_files)}个条目"
        
        logger.info(f"✓ 成功索引 {len(result)} 个PDF文件: {result}")
        return True
        
    except Exception as e:
        logger.error(f"✗ index_folder 测试失败: {e}")
        return False


def test_ingest_paper():
    """测试论文摄取功能"""
    logger.info("=== 测试 ingest_paper 功能 ===")
    
    # 检查是否有已索引的论文
    if not papers_index:
        logger.warning("没有已索引的论文，请先运行 test_index_folder")
        return False
    
    # 取第一个已索引的论文进行测试
    paper_uri = list(papers_index.keys())[0]
    logger.info(f"测试论文: {paper_uri}")
    
    try:
        # 测试摄取功能
        result = server.ingest_paper(paper_uri)
        
        # 验证结果
        assert isinstance(result, str), "返回结果应该是字符串路径"
        cache_path = pathlib.Path(result)
        assert cache_path.exists(), f"缓存文件应该存在: {cache_path}"
        
        # 验证缓存内容
        cached_content = cache_path.read_text(encoding="utf-8")
        assert len(cached_content) > 0, "缓存内容不应为空"
        
        logger.info(f"✓ 成功摄取论文，缓存路径: {result}")
        logger.info(f"✓ 缓存内容长度: {len(cached_content)} 字符")
        return True
        
    except Exception as e:
        logger.error(f"✗ ingest_paper 测试失败: {e}")
        return False


def test_extract_entities():
    """测试实体提取功能（真实LLM调用）"""
    logger.info("=== 测试 extract_entities 功能 ===")
    
    # 检查是否有已索引的论文
    if not papers_index:
        logger.warning("没有已索引的论文，请先运行 test_index_folder")
        return False
    
    # 取第一个已索引的论文进行测试
    paper_uri = list(papers_index.keys())[0]
    logger.info(f"测试论文: {paper_uri}")
    
    # 确保论文文本已缓存
    txt_cache = server._cache_path(paper_uri, ".txt")
    if not txt_cache.exists():
        logger.info("论文文本未缓存，先进行摄取...")
        try:
            server.ingest_paper(paper_uri)
        except Exception as e:
            logger.error(f"摄取论文失败: {e}")
            return False
    
    try:
        # 测试实体提取（真实LLM调用）
        logger.info("开始提取实体（这可能需要一些时间...）")
        result = server.extract_entities(paper_uri)
        
        # 验证结果
        assert isinstance(result, list), "返回结果应该是列表"
        assert len(result) >= 0, "结果应该是有效的材料列表"
        
        if len(result) > 0:
            assert isinstance(result[0], Material), "结果应该是Material对象"
            logger.info(f"✓ 成功提取 {len(result)} 个材料实体")
            
            # 显示第一个材料的详细信息
            first_material = result[0]
            logger.info(f"✓ 第一个材料: {first_material.id} - {first_material.name}")
            
            if first_material.lpbf_process:
                logger.info(f"✓ 包含LPBF工艺参数")
                if first_material.lpbf_process.energy_density_J_mm3:
                    logger.info(f"✓ 能量密度: {first_material.lpbf_process.energy_density_J_mm3:.2f} J/mm³")
            
            if first_material.properties:
                logger.info(f"✓ 包含 {len(first_material.properties)} 个材料属性")
        else:
            logger.info("✓ 未提取到材料实体（可能论文不包含相关内容）")
        
        # 验证实体缓存文件
        entities_cache = server._cache_path(paper_uri, "_entities.json")
        assert entities_cache.exists(), "应该创建实体缓存文件"
        logger.info(f"✓ 实体缓存文件已创建: {entities_cache}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ extract_entities 测试失败: {e}")
        return False


def run_tests():
    """运行所有测试"""
    logger.info("=" * 50)
    logger.info("开始 PapersKG 核心功能测试")
    logger.info("=" * 50)
    
    tests = [
        ("index_folder", test_index_folder),
        ("ingest_paper", test_ingest_paper), 
        ("extract_entities", test_extract_entities),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n开始测试: {test_name}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            logger.info(f"✓ {test_name} 测试通过")
        else:
            logger.error(f"✗ {test_name} 测试失败")
    
    # 统计结果
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info("测试结果统计:")
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n总计: {passed}/{total} 通过")
    logger.info(f"成功率: {passed/total*100:.1f}%")
    logger.info("=" * 50)
    
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)