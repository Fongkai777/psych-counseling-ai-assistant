from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "github入门-项目发布指南.pdf"
IMAGE_DIR = ROOT / "docs" / "assets" / "github"


def register_fonts():
    pdfmetrics.registerFont(
        TTFont("ArialUnicode", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    )


def style_sheet():
    styles = getSampleStyleSheet()
    base_font = "ArialUnicode"
    styles.add(
        ParagraphStyle(
            name="CnTitle",
            fontName=base_font,
            fontSize=24,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f2a44"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnSubtitle",
            fontName=base_font,
            fontSize=11,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#566176"),
            spaceAfter=22,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnH1",
            fontName=base_font,
            fontSize=17,
            leading=23,
            textColor=colors.HexColor("#24324b"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnH2",
            fontName=base_font,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#2f3e5c"),
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnBody",
            fontName=base_font,
            fontSize=10.5,
            leading=17,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#202733"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnSmall",
            fontName=base_font,
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#5d6677"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnBullet",
            fontName=base_font,
            fontSize=10,
            leading=16,
            textColor=colors.HexColor("#202733"),
        )
    )
    return styles


def p(text, styles, name="CnBody"):
    return Paragraph(text, styles[name])


def bullets(items, styles):
    return ListFlowable(
        [ListItem(p(item, styles, "CnBullet"), leftIndent=8) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
        bulletFontName="ArialUnicode",
        bulletFontSize=8,
    )


def code(text):
    return Preformatted(
        text,
        ParagraphStyle(
            name="Code",
            fontName="Courier",
            fontSize=8.6,
            leading=12,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f4f6fa"),
            borderColor=colors.HexColor("#d9dee8"),
            borderWidth=0.5,
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=9,
        ),
    )


def table(rows, col_widths):
    data = [[Paragraph(cell, ParagraphStyle(name="Cell", fontName="ArialUnicode", fontSize=9.5, leading=14)) for cell in row] for row in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2a44")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfd6e4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def screenshot(filename, max_width=16.2 * cm, max_height=12.5 * cm):
    path = IMAGE_DIR / filename
    img = Image(str(path))
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"
    return img


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("ArialUnicode", 8)
    canvas.setFillColor(colors.HexColor("#7a8394"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "GitHub 入门 - AI 项目发布指南")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def build():
    register_fonts()
    styles = style_sheet()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.7 * cm,
        title="GitHub 入门 - 项目发布指南",
        author="Codex",
    )

    story = []
    story.append(p("GitHub 入门：把你的 AI 项目安全发布出去", styles, "CnTitle"))
    story.append(p("写给第一次使用 GitHub 的你：少一点概念恐吓，多一点可执行步骤。", styles, "CnSubtitle"))

    story.append(p("1. 你要先建立的心智模型", styles, "CnH1"))
    story.append(p("Git 和 GitHub 不是同一个东西。Git 是本地的版本管理工具，像项目的时间机器；GitHub 是云端代码托管平台，像把项目放到一个可以展示、协作和备份的地方。", styles))
    story.append(
        bullets(
            [
                "<b>本地仓库</b>：你电脑里的项目文件夹，被 Git 管理之后就叫本地仓库。",
                "<b>远程仓库</b>：GitHub 上的项目空间。",
                "<b>commit</b>：一次明确的存档，最好配一句说明。",
                "<b>push</b>：把本地 commit 上传到 GitHub。",
                "<b>pull</b>：把 GitHub 上的新变化拉回本地。",
            ],
            styles,
        )
    )

    story.append(p("2. 你的这个项目哪些文件不能上传", styles, "CnH1"))
    story.append(p("AI/RAG 项目比普通项目更容易混入隐私数据。上传前先分清：代码可以公开，运行数据和密钥不能公开。", styles))
    rows = [
        ["文件或目录", "是否上传", "原因"],
        [".env", "不要上传", "里面有真实 API key，泄露后别人可能消耗你的额度。"],
        ["data/", "不要上传", "里面通常有 SQLite 数据库、知识库原文、个人语气文档、人工终稿。"],
        [".venv/", "不要上传", "这是你本机的 Python 环境，体积大，别人也不能直接复用。"],
        [".idea/", "不要上传", "PyCharm 本地配置，不属于项目核心代码。"],
        [".DS_Store", "不要上传", "macOS 自动生成的系统文件。"],
        [".env.example", "可以上传", "只放配置字段，不放真实 key，别人照着复制即可。"],
        ["requirements.txt", "必须上传", "别人用它安装依赖。"],
    ]
    story.append(table(rows, [3.2 * cm, 2.5 * cm, 10.5 * cm]))

    story.append(p("3. 第一次上传：完整流程", styles, "CnH1"))
    story.append(p("下面这些命令在 PyCharm Terminal 或系统终端里执行。先进入项目目录：", styles))
    story.append(code("cd /Users/changfengkai/PycharmProjects/PythonProject\npwd"))
    story.append(p("初始化 Git 仓库：", styles))
    story.append(code("git init"))
    story.append(p("检查哪些文件会被 Git 看到：", styles))
    story.append(code("git status\ngit status --ignored"))
    story.append(p("安全添加文件。第一次不要用 git add .，先明确列出要上传的代码文件：", styles))
    story.append(
        code(
            "git add README.md .gitignore .env.example requirements.txt app.py streamlit_app.py \\\n"
            "  psych_ai_assistant scripts web"
        )
    )
    story.append(p("再次检查。确认不要出现 .env、data/、.venv/、.idea/：", styles))
    story.append(code("git status"))
    story.append(p("提交一次本地版本：", styles))
    story.append(code('git commit -m "Initial AI content operations assistant"'))

    story.append(PageBreak())
    story.append(p("4. 在 GitHub 创建远程仓库", styles, "CnH1"))
    story.append(screenshot("github-new-repository.png", max_height=10.8 * cm))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        bullets(
            [
                "打开 https://github.com/new",
                "Repository name 可以写：psych-ai-content-assistant",
                "建议第一次先选 Private，等确认无隐私文件后再考虑 Public。",
                "不要勾选 Add a README file。",
                "不要勾选 .gitignore。",
                "不要勾选 license，后面想清楚再加。",
            ],
            styles,
        )
    )
    story.append(p("创建完成后，GitHub 会给你类似下面的命令。这里已经换成你现在这个仓库的真实地址：", styles))
    story.append(
        code(
            "git remote add origin https://github.com/Fongkai777/psych-ai-content-assistant.git\n"
            "git branch -M main\n"
            "git push -u origin main"
        )
    )

    story.append(p("5. 推送成功后你会看到什么", styles, "CnH1"))
    story.append(p("推送成功后，打开仓库首页会看到文件列表和 README。README 会直接显示在仓库首页，所以它很像这个项目的门面。", styles))
    story.append(screenshot("github-repository-home-top.png", max_height=11.8 * cm))
    story.append(Spacer(1, 0.2 * cm))
    story.append(p("你现在这个项目首页里已经能看到代码结构、启动方式、功能说明和隐私安全说明。以后投递简历时，面试官大概率会先看这里。", styles))

    story.append(p("6. 以后每次更新项目怎么做", styles, "CnH1"))
    story.append(p("日常更新只需要四步：看状态、添加文件、提交、推送。", styles))
    story.append(
        code(
            "git status\n"
            "git add README.md streamlit_app.py psych_ai_assistant/prompts.py\n"
            "git commit -m \"Update answer workflow\"\n"
            "git push"
        )
    )
    story.append(p("注意：还是不要习惯性使用 git add .。等你熟练之后可以用，但第一次做简历项目，稳比快重要。", styles))

    story.append(p("7. 常见概念速查", styles, "CnH1"))
    rows = [
        ["概念", "你可以怎么理解"],
        ["git status", "看看现在有哪些文件变了、哪些准备提交。"],
        ["git add", "把文件放进这次提交的购物车。"],
        ["git commit", "给购物车里的文件拍一张版本快照。"],
        ["git push", "把本地快照上传到 GitHub。"],
        ["git pull", "把 GitHub 上的新内容同步到本地。"],
        ["branch", "一条独立开发线。先不用深究，默认 main 就够。"],
        ["README.md", "项目首页说明，面试官通常先看它。"],
        [".gitignore", "告诉 Git 哪些文件不要管。"],
    ]
    story.append(table(rows, [3.5 * cm, 12.7 * cm]))

    story.append(p("8. 这个项目的公开版建议", styles, "CnH1"))
    story.append(
        bullets(
            [
                "README 重点讲清楚：意图识别、RAG、人工反馈闭环、私有知识库、本地隐私保护。",
                "不要上传真实 PDF、真实日记、真实回答和 SQLite 数据库。",
                "可以加 samples/ 示例文件，但必须是虚构或彻底脱敏内容。",
                "如果要给面试官演示，可以本地启动 Streamlit，或者录一段短视频。",
                "如果要公开仓库，先让别人或 AI 帮你再扫一遍敏感信息。",
            ],
            styles,
        )
    )

    story.append(p("9. 出错时先看这几个问题", styles, "CnH1"))
    story.append(
        bullets(
            [
                "push 要求登录：先确认你有没有配置 GitHub token 或 GitHub CLI。",
                "remote already exists：说明远程地址已经添加过，可以用 git remote -v 查看。",
                "nothing to commit：说明当前没有新改动，或者你还没 git add。",
                "accidentally staged wrong file：用 git restore --staged 文件名 取消暂存。",
                "误提交了 .env：不要只删文件，要立即撤销 key，并重写 Git 历史或重新建仓库。",
            ],
            styles,
        )
    )

    story.append(p("10. 你现在最应该记住的三句话", styles, "CnH1"))
    story.append(
        bullets(
            [
                "GitHub 展示的是项目能力，不需要展示你的真实私有数据。",
                "先 git status，再 git add，最后 git commit，不要闭眼上传。",
                "API key 泄露后第一反应不是删除 GitHub 文件，而是立刻去平台撤销这个 key。",
            ],
            styles,
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
